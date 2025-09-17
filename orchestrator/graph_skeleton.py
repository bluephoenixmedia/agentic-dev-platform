# This script defines a multi-agent workflow using LangGraph for an automated software development process.
# Each agent is a node in the graph, and the state is passed between them.
# The workflow is designed to be design-doc-first, where a canonical design document guides the process.

# --- Core Imports ---
import os
import re
import json
import signal
import traceback
from typing import TypedDict, List, Dict, Optional, Literal

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# --- LangChain Imports for LLM Integration ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.callbacks import StreamingStdOutCallbackHandler

# --- Custom Tool Imports ---
from tools.shell_tool import run_shell_command

# === 1. State Definition ===
class State(TypedDict, total=False):
    """ The complete state of the agentic development workflow. """
    design_doc: Optional[str]
    roadmap: Optional[List[Dict]]
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
    """ Agent responsible for loading and maintaining the design document. """
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
    """ Agent that uses an LLM to generate a development roadmap. """
    print("---EXECUTING PLANNER---")
    log = state.get("run_log", [])
    
    # Check for a persistent roadmap file to enable resumable runs.
    if os.path.exists("roadmap.json"):
        log.append("[Planner] Info: Existing roadmap.json found. Loading to resume progress.")
        with open("roadmap.json", "r") as f:
            roadmap = json.load(f)
        
        # Resumability Logic: Check workspace for completed tasks and update their status.
        workspace_files = [f for f in os.listdir("/workspaces/agentic-dev-platform/workspace") if f.endswith(".md")]
        for phase in roadmap:
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
                System: You are an expert project manager. Your task is to analyze a software design document and create a detailed, machine-readable roadmap in JSON format.
                The roadmap must be a list of phases, and each phase must contain a list of tasks.
                
                Here is an example of the required JSON structure:
                {{
                  "phases": [
                    {{
                      "phase": "Phase 0: Foundations",
                      "tasks": [
                        {{
                          "id": "p0-t1",
                          "title": "Setup Docker Environment",
                          "kind": "ops",
                          "status": "todo"
                        }}
                      ]
                    }}
                  ]
                }}

                Human: Here is the design document:
                {design_doc}
                """,
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
                final_roadmap = roadmap_output.get("phases", roadmap_output.get("roadmap", []))
            elif isinstance(roadmap_output, list):
                final_roadmap = roadmap_output
            else:
                final_roadmap = []

            with open("roadmap.json", "w") as f:
                json.dump(final_roadmap, f, indent=2)

            num_tasks = sum(len(phase.get("tasks", [])) for phase in final_roadmap if isinstance(phase, dict))
            log.append(f"[Planner] Success: LLM generated a roadmap with {len(final_roadmap)} phases and {num_tasks} total tasks.")
            return {**state, "roadmap": final_roadmap, "run_log": log}
        except Exception as e:
            log.append(f"[Planner] Failure: LLM roadmap generation failed. {e}")
            return {**state, "error": f"LLM roadmap generation failed: {e}", "run_log": log}
    else:
        log.append("[Planner] Failure: Design document missing from state.")
        return {**state, "error": "Design document missing for planner", "run_log": log}

def architect(state: State) -> State:
    """ Agent that reviews the roadmap and selects the next available task. """
    print("---EXECUTING ARCHITECT---")
    log = state.get("run_log", [])
    roadmap = state.get("roadmap", [])
    if not roadmap:
        log.append("[Architect] Info: Roadmap is empty. No tasks to select.")
        return {**state, "current_item": None, "run_log": log}

    next_task = None
    selected_phase_name = "N/A"
    
    for phase in roadmap:
        if isinstance(phase, dict):
            for task in phase.get("tasks", []):
                if task.get("status") == "todo":
                    next_task = task
                    selected_phase_name = phase.get("phase", "N/A")
                    break
        if next_task:
            break
    
    if next_task:
        log.append(f"[Architect] Success: Selected task '{next_task.get('id', 'N/A')}' from phase '{selected_phase_name}' for development.")
        return {**state, "current_item": next_task, "run_log": log}
    else:
        log.append("[Architect] Info: All tasks on the roadmap are complete.")
        return {**state, "current_item": None, "run_log": log}

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
        # NEW: A more advanced prompt that asks the LLM for a structured JSON output
        # containing both a natural language plan and a list of shell commands.
        prompt = PromptTemplate(
            template="""
            System: You are an expert software developer. Your task is to create a plan to accomplish a given task, and then provide the exact shell commands needed to execute that plan.
            You must return your response as a single JSON object with two keys: "plan" and "commands".
            The "plan" should be a natural language description of the steps.
            The "commands" should be a list of executable shell command strings.
            
            Example:
            {{
                "plan": "First, I will create a new directory for the feature. Then, I will create a placeholder file inside it.",
                "commands": [
                    "mkdir -p /workspaces/agentic-dev-platform/workspace/new_feature",
                    "touch /workspaces/agentic-dev-platform/workspace/new_feature/index.js"
                ]
            }}

            Design Document Context:
            {design_doc}
            
            Human: Please provide the plan and commands for the task: "{task_title}".
            """,
            input_variables=["task_title", "design_doc"],
        )
        
        print(f"\n--- Coder LLM Stream for Task: {current_item['title']} ---")
        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL"), 
            base_url=os.getenv("OLLAMA_BASE_URL"),
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        # We need to clean the LLM output as it sometimes includes markdown backticks for JSON
        # and then parse it.
        parser = JsonOutputParser()
        chain = prompt | llm | (lambda x: json.loads(re.search(r'```json\n(.*?)\n```', x.content, re.DOTALL).group(1))) | parser
        
        response = chain.invoke({
            "task_title": current_item['title'],
            "design_doc": state['design_doc']
        })
        print("--- End of Stream ---\n")

        implementation_plan = response.get("plan", "No plan generated.")
        commands_to_execute = response.get("commands", [])
        
        # Save the natural language plan to a file, as before.
        task_id = current_item.get('id', 'unnamed_task')
        filename = f"{task_id.replace(' ', '_').lower()}.md"
        filepath = f"/workspaces/agentic-dev-platform/workspace/{filename}"
        with open(filepath, "w") as f:
            f.write(f"# Implementation Plan for: {current_item['title']}\n\n{implementation_plan}")
        log.append(f"[Coder] Success: Generated and wrote implementation plan to {filepath}")
        
        # NEW: Execute the commands extracted from the LLM's response.
        if commands_to_execute:
            log.append(f"[Coder] Info: Executing {len(commands_to_execute)} commands...")
            for cmd in commands_to_execute:
                exit_code, output = run_shell_command(cmd)
                log.append(f"[Coder] Command '{cmd}':\n--- Exit Code: {exit_code}\n--- Output:\n{output}\n---")
                if exit_code != 0:
                    log.append(f"[Coder] Failure: Command '{cmd}' failed. Halting execution for this task.")
                    # Setting the error state will allow for future error-handling logic.
                    return {**state, "error": f"Command failed: {cmd}", "run_log": log}
        else:
            log.append("[Coder] Info: No commands were generated by the LLM for this task.")

        # Update the task status in the central roadmap.
        updated_roadmap = state["roadmap"]
        task_found_and_updated = False
        for phase in updated_roadmap:
            if isinstance(phase, dict):
                for task in phase.get("tasks", []):
                    if task.get("id") == current_item.get("id"):
                        task["status"] = "done"
                        task_found_and_updated = True
                        break
            if task_found_and_updated:
                break
        
        log.append(f"[Coder] Info: Marked task '{task_id}' as 'done' in the roadmap.")
        return {**state, "repo_changes": [f"EXECUTED_PLAN: {filepath}"], "roadmap": updated_roadmap, "run_log": log}

    except Exception as e:
        log.append(f"[Coder] Failure: Coder agent failed for task {current_item.get('id', 'unknown')}. Details: {e}")
        return {**state, "error": f"Coder agent failed: {e}", "run_log": log}

def tester(state: State) -> State:
    """ Agent that simulates running tests on the changes made by the Coder. """
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
    """ Agent that is triggered on failure to analyze logs. """
    print("---EXECUTING LOG ANALYST---")
    log = state.get("run_log", [])
    error_message = state.get('error', 'Unknown error')
    log.append(f"[LogAnalyst] Info: An error was detected. Analyzing details: {error_message}")
    return {**state, "run_log": log}

# === 3. Conditional Edges ===
def should_continue(state: State) -> Literal["Architect", "__end__"]:
    """ Determines if the main workflow should continue to the Architect or end. """
    if state.get("current_item") is not None:
        return "Architect"
    else:
        return "__end__"

# === 4. Graph Definition and Compilation ===
memory = SqliteSaver.from_conn_string(":memory:")
builder = StateGraph(State)
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
builder.add_edge("LogAnalyst", "Coder")
graph = builder.compile(checkpointer=memory)
print("Graph compiled successfully.")

# === 5. Main Execution Block ===
if __name__ == "__main__":
    
    # --- Graceful Shutdown Handler ---
    # This allows the user to press Ctrl+C to stop the workflow cleanly.
    shutdown_flag = [False]
    def signal_handler(sig, frame):
        print("\n---Shutdown signal received. Allowing current agent to finish...---")
        shutdown_flag[0] = True
    signal.signal(signal.SIGINT, signal_handler)

    print("\n---Invoking agent workflow (Press Ctrl+C to save progress and exit)---")
    
    initial_state = {"run_log": [], "metadata": {"project": "agentic-dev-platform"}}
    config = {"configurable": {"thread_id": "proj-thread-2"}, "recursion_limit": 250}
    final_state = {}

    try:
        # We use `stream` instead of `invoke` to process events one by one,
        # which allows us to check for the shutdown flag between steps.
        for event in graph.stream(initial_state, config=config):
            final_state = event
            if shutdown_flag[0]:
                print("---Workflow interrupted. Saving final state...---")
                break
    finally:
        # --- Save Final State and Log ---
        # This block ensures that the final state is always saved,
        # even if an error occurs or the user interrupts the process.
        print("\n---Workflow Complete or Interrupted---")
        
        final_roadmap = final_state.get(list(final_state.keys())[0], {}).get("roadmap")
        if final_roadmap:
            with open("roadmap.json", "w") as f:
                json.dump(final_roadmap, f, indent=2)
            print("Final roadmap state has been saved to roadmap.json")
            
        final_log = final_state.get(list(final_state.keys())[0], {}).get("run_log", ["No log entries found."])
        with open("run_log.txt", "w") as f:
            f.write("AGENTIC DEV PLATFORM - RUN LOG\n=================================\n\n")
            f.write("\n".join(final_log))
        print("Final run log has been written to run_log.txt")

