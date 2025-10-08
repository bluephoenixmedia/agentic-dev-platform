# This script defines a multi-agent workflow using LangGraph for an automated software development process.
# Each agent is a node in the graph, and the state is passed between them.
# The workflow is designed to be design-doc-first, where a canonical design document guides the process.

# --- Core Imports ---
import os
import re
import json
import signal
import sys
import traceback
from typing import TypedDict, List, Dict, Optional, Literal

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# --- LangChain Imports for LLM Integration ---
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import BaseMessage

# --- Custom Tool Imports ---
from tools.shell_tool import run_shell_command

# === 1. State Definition ===
class State(TypedDict, total=False):
    """ The complete state of the agentic development workflow. """
    design_doc: Optional[str]
    roadmap: Optional[Dict]
    current_item: Optional[Dict]
    repo_changes: Optional[List[str]]
    run_status: Optional[str]
    logs_tail: Optional[str]
    approvals: Optional[Dict[str, bool]]
    metadata: Optional[Dict[str, str]]
    error: Optional[str]
    run_log: Optional[List[str]]

# === 2. Agent Definitions ===

def doc_agent(state: State) -> State:
    """ Agent responsible for loading the design document. """
    print("---EXECUTING DOC AGENT---")
    log = state.get("run_log", [])
    try:
        doc_path = "/workspaces/agentic-dev-platform/docs/Design_Document.md"
        with open(doc_path, 'r') as f:
            design_doc = f.read()
        log.append(f"[DocAgent] Success: Loaded Design_Document.md ({len(design_doc)} chars)")
        return {**state, "design_doc": design_doc, "run_log": log}
    except FileNotFoundError:
        log.append(f"[DocAgent] Failure: Design document not found at {doc_path}")
        return {**state, "error": "Design document not found", "run_log": log}

def planner(state: State) -> State:
    """ Agent that uses an LLM to generate or load a development roadmap. """
    print("---EXECUTING PLANNER---")
    log = state.get("run_log", [])
    roadmap_path = "/workspaces/agentic-dev-platform/roadmap.json"
    
    if os.path.exists(roadmap_path):
        log.append("[Planner] Info: Existing roadmap.json found. Loading to resume progress.")
        with open(roadmap_path, 'r') as f:
            roadmap = json.load(f)
        
        workspace_files = [f for f in os.listdir("/workspaces/agentic-dev-platform/workspace") if f.endswith(".md")]
        for phase in roadmap.get("phases", []):
            if isinstance(phase, dict):
                for task in phase.get("tasks", []):
                    task_filename = f"{task['id'].replace(' ', '_').lower()}.md"
                    if task_filename in workspace_files:
                        task["status"] = "done"
        
        return {**state, "roadmap": roadmap, "run_log": log}

    if state.get("design_doc"):
        try:
            prompt = PromptTemplate(
                template="""
                System: You are an expert project manager...Create a roadmap...
                Human: Here is the design document:\n{design_doc}""",
                input_variables=["design_doc"],
            )
            
            print("\n--- Planner LLM Stream ---")
            llm = ChatOllama(
                model=os.getenv("OLLAMA_MODEL"), 
                base_url=os.getenv("OLLAMA_BASE_URL"),
                callbacks=[StreamingStdOutCallbackHandler()]
            )
            parser = JsonOutputParser()
            chain = prompt | llm | parser
            roadmap_output = chain.invoke({"design_doc": state["design_doc"]})
            print("--- End of Stream ---\n")
            
            if isinstance(roadmap_output, dict):
                final_roadmap = roadmap_output
            else:
                # This handles cases where the LLM might return a raw string that needs parsing.
                match = re.search(r'\{.*\}', roadmap_output, re.DOTALL)
                if match:
                    final_roadmap = json.loads(match.group(0))
                else:
                    raise ValueError("No valid JSON object found in LLM output.")

            num_tasks = sum(len(phase.get("tasks", [])) for phase in final_roadmap.get("phases", []) if isinstance(phase, dict))
            log.append(f"[Planner] Success: LLM generated a new roadmap with {len(final_roadmap.get('phases', []))} phases and {num_tasks} total tasks.")
            
            with open(roadmap_path, "w") as f:
                json.dump(final_roadmap, f, indent=2)
            log.append(f"[Planner] Info: Roadmap saved to {roadmap_path}")
            
            return {**state, "roadmap": final_roadmap, "run_log": log}
        except Exception as e:
            log.append(f"[Planner] Failure: LLM roadmap generation failed. {e}")
            return {**state, "error": f"LLM roadmap generation failed: {e}", "run_log": log}
    else:
        log.append("[Planner] Failure: Design document missing.")
        return {**state, "error": "Design document missing", "run_log": log}


def architect(state: State) -> State:
    """ Agent that selects the next available task from the roadmap. """
    print("---EXECUTING ARCHITECT---")
    log = state.get("run_log", [])
    roadmap = state.get("roadmap")

    if not roadmap or not isinstance(roadmap, dict):
        log.append("[Architect] Info: Roadmap is missing or invalid. No tasks to select.")
        return {**state, "current_item": None, "run_log": log}

    next_task = None
    
    for phase in roadmap.get("phases", []):
        for task in phase.get("tasks", []):
            if task.get("status") == "todo":
                next_task = task
                log.append(f"[Architect] Success: Selected task '{next_task.get('id')}' from phase '{phase.get('phase')}' for development.")
                return {**state, "current_item": next_task, "run_log": log}
    
    log.append("[Architect] Info: All tasks on the roadmap are complete.")
    return {**state, "current_item": None, "run_log": log}

def extract_json_from_response(message: BaseMessage) -> str:
    """
    Extracts a JSON object from the LLM's response content, which may include
    conversational text and markdown code blocks.
    """
    text = message.content
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return "{}"

def coder(state: State) -> State:
    """
    Agent that generates and executes a plan to complete the current task.
    """
    print("---EXECUTING CODER---")
    log = state.get("run_log", [])
    current_item = state.get("current_item")
    if not current_item:
        log.append("[Coder] Info: No current task to execute.")
        return {**state, "run_log": log}

    try:
        prompt = PromptTemplate(
            template="""
            System: You are an expert software developer...Return a single JSON object...
            Human: Please provide the plan and commands for the task: "{task_title}".
            Design Document Context:\n{design_doc}""",
            input_variables=["task_title", "design_doc"],
        )
        
        print(f"\n--- Coder LLM Stream for Task: {current_item['title']} ---")
        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL"), 
            base_url=os.getenv("OLLAMA_BASE_URL"),
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        parser = JsonOutputParser()
        chain = prompt | llm | extract_json_from_response | parser
        
        response = chain.invoke({
            "task_title": current_item['title'],
            "design_doc": state['design_doc']
        })
        print("--- End of Stream ---\n")

        implementation_plan = response.get("plan", "No plan generated.")
        commands_to_execute = response.get("commands", [])
        
        task_id = current_item.get('id', 'unnamed_task')
        filename = f"{task_id.replace(' ', '_').lower()}.md"
        filepath = f"/workspaces/agentic-dev-platform/workspace/{filename}"
        with open(filepath, "w") as f:
            f.write(f"# Implementation Plan for: {current_item['title']}\n\n{implementation_plan}")
        log.append(f"[Coder] Success: Generated and wrote implementation plan to {filepath}")
        
        changes = [f"CREATED: {filepath}"]
        
        if commands_to_execute:
            log.append(f"[Coder] Info: Executing {len(commands_to_execute)} commands...")
            for cmd in commands_to_execute:
                exit_code, output = run_shell_command(cmd)
                log.append(f"[Coder] Command '{cmd}':\n--- Exit Code: {exit_code}\n--- Output:\n{output}\n---")
                if exit_code != 0:
                    log.append(f"[Coder] Failure: Command '{cmd}' failed. Halting execution.")
                    return {**state, "error": f"Command failed: {cmd}", "run_log": log}
        else:
            log.append("[Coder] Info: No commands were generated by the LLM for this task.")

        updated_roadmap = state["roadmap"]
        for phase in updated_roadmap.get("phases", []):
            for task in phase.get("tasks", []):
                if task.get("id") == current_item.get("id"):
                    task["status"] = "done"
                    break
        
        log.append(f"[Coder] Info: Marked task '{task_id}' as 'done' in the roadmap.")
        return {**state, "repo_changes": changes, "roadmap": updated_roadmap, "run_log": log}

    except Exception as e:
        log.append(f"[Coder] Failure: Coder agent failed. Details: {e}")
        traceback.print_exc()
        return {**state, "error": f"Coder agent failed: {e}", "run_log": log}

def tester(state: State) -> State:
    """ Agent that simulates running tests. """
    print("---EXECUTING TESTER---")
    log = state.get("run_log", [])
    repo_changes = state.get("repo_changes")
    if repo_changes:
        log.append(f"[Tester] Success: Simulated tests PASSED for changes: {', '.join(repo_changes)}")
        return {**state, "repo_changes": [], "run_log": log}
    else:
        log.append("[Tester] Info: No file changes detected to test.")
        return {**state, "run_log": log}

def cicd(state: State) -> State:
    """ Agent that simulates a CI/CD pipeline. """
    print("---EXECUTING CICD---")
    log = state.get("run_log", [])
    current_item = state.get("current_item")
    if current_item and current_item.get("status") == "done":
         log.append(f"[CICD] Success: Simulated build for task '{current_item.get('title', 'N/A')}'.")
    else:
        log.append(f"[CICD] Info: No completed task to process.")
    return {**state, "run_log": log}

def log_analyst(state: State) -> State:
    """ Agent that is triggered on failure. """
    print("---EXECUTING LOG ANALYST---")
    log = state.get("run_log", [])
    error_message = state.get('error', 'Unknown error')
    log.append(f"[LogAnalyst] Info: An error was detected. Analyzing details: {error_message}")
    return state

# === 3. Graph Definition and Control Flow ===
def should_continue(state: State) -> Literal["Architect", "__end__"]:
    """ Determines if the main workflow should continue to the Architect or end. """
    if state.get("current_item") is not None:
        return "Architect"
    else:
        return "__end__"

memory = SqliteSaver.from_conn_string(":memory:")
builder = StateGraph(State)

# NEW: Uncommented the node definitions to correctly build the graph.
builder.add_node("DocAgent", doc_agent)
builder.add_node("Planner", planner)
builder.add_node("Architect", architect)
builder.add_node("Coder", coder)
builder.add_node("Tester", tester)
builder.add_node("CICD", cicd)
builder.add_node("LogAnalyst", log_analyst)

builder.set_entry_point("DocAgent")
builder.add_edge("DocAgent", "Planner")
builder.add_edge("Planner", "Architect")
builder.add_edge("Architect", "Coder")
builder.add_edge("Coder", "Tester")
builder.add_edge("Tester", "CICD")
builder.add_conditional_edges("CICD", should_continue)
builder.add_edge("LogAnalyst", "Coder") # Simplified error handling loop

graph = builder.compile(checkpointer=memory)
print("Graph compiled successfully.")

# === 4. Graceful Shutdown and Main Execution ===
current_state = {}
def save_and_exit(signum, frame):
    """ Signal handler for graceful shutdown. """
    print("\n---Shutdown signal received. Saving state and exiting.---")
    if 'roadmap' in current_state:
        roadmap_path = "/workspaces/agentic-dev-platform/roadmap.json"
        with open(roadmap_path, "w") as f:
            json.dump(current_state['roadmap'], f, indent=2)
        print(f"Latest roadmap progress saved to {roadmap_path}")
    sys.exit(0)

signal.signal(signal.SIGINT, save_and_exit)

if __name__ == "__main__":
    print("\n---Invoking agent workflow (Press Ctrl+C to save progress and exit)---")
    initial_state = {"run_log": [], "metadata": {"project": "agentic-dev-platform"}}
    config = {"configurable": {"thread_id": "proj-thread-4"}, "recursion_limit": 250}
    final_state_from_run = {}

    try:
        for event in graph.stream(initial_state, config=config):
            current_state = list(event.values())[0]
        final_state_from_run = current_state
    finally:
        print("\n---Workflow Complete or Interrupted---")
        
        final_roadmap = final_state_from_run.get("roadmap")
        if final_roadmap:
            with open("/workspaces/agentic-dev-platform/roadmap.json", "w") as f:
                json.dump(final_roadmap, f, indent=2)
            print("Final roadmap state has been saved to roadmap.json")
            
        final_log = final_state_from_run.get("run_log", ["No log entries found."])
        with open("/workspaces/agentic-dev-platform/run_log.txt", "w") as f:
            f.write("AGENTIC DEV PLATFORM - RUN LOG\n=================================\n\n")
            f.write("\n".join(final_log))
        print("Final run log has been written to run_log.txt")

