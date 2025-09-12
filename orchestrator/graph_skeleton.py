# This script defines a multi-agent workflow using LangGraph for an automated software development process.
# Each agent is a node in the graph, and the state is passed between them.
# The workflow is designed to be design-doc-first, where a canonical design document guides the process.

# --- Core Imports ---
import os
import json
from typing import TypedDict, List, Dict, Optional, Literal

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# --- LangChain Imports for LLM Integration ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# --- Custom Tool Imports ---
# We now import our new shell command execution tool.
from tools.shell_tool import run_shell_command


# === 1. State Definition ===
class State(TypedDict, total=False):
    """
    The complete state of the agentic development workflow.
    """
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
    """
    Agent responsible for loading and maintaining the design document.
    """
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
    """
    Agent that uses an LLM to generate a development roadmap from the design document.
    """
    print("---EXECUTING PLANNER---")
    log = state.get("run_log", [])
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
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", convert_system_message_to_human=True)
            parser = JsonOutputParser()
            chain = prompt | llm | parser
            roadmap_output = chain.invoke({"design_doc": state["design_doc"]})
            
            if isinstance(roadmap_output, dict):
                final_roadmap = roadmap_output.get("phases", roadmap_output.get("roadmap", []))
            elif isinstance(roadmap_output, list):
                final_roadmap = roadmap_output
            else:
                final_roadmap = []

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
    """
    Agent that reviews the roadmap and selects the next available task.
    """
    print("---EXECUTING ARCHITECT---")
    log = state.get("run_log", [])
    roadmap = state.get("roadmap", [])
    if not roadmap:
        log.append("[Architect] Info: Roadmap is empty. No tasks to select.")
        return {**state, "current_item": None, "run_log": log}

    next_task = None
    for item in roadmap:
        if isinstance(item, dict) and item.get("status") == "todo" and "tasks" not in item:
            next_task = item
            break
    
    if not next_task:
        for phase in roadmap:
            if isinstance(phase, dict):
                for task in phase.get("tasks", []):
                    if task.get("status") == "todo":
                        next_task = task
                        break
            if next_task:
                break
    
    if next_task:
        log.append(f"[Architect] Success: Selected task '{next_task.get('id', 'N/A')}' for development.")
        return {**state, "current_item": next_task, "run_log": log}
    else:
        log.append("[Architect] Info: All tasks on the roadmap are complete.")
        return {**state, "current_item": None, "run_log": log}


def coder(state: State) -> State:
    """
    Agent responsible for implementing the current task by generating a plan and executing commands.
    """
    print("---EXECUTING CODER---")
    log = state.get("run_log", [])
    current_item = state.get("current_item")
    if not current_item:
        log.append("[Coder] Info: No current task to execute.")
        return {**state, "run_log": log}

    try:
        # --- Step 1: Generate Implementation Plan (as before) ---
        prompt = PromptTemplate(
            template="""
            System: You are an expert software developer. Your current task is to "{task_title}".
            Based on the design document provided, create a high-level, step-by-step plan for how a human developer would accomplish this task.
            Design Document Context:
            {design_doc}
            Human: Please provide the plan for the task: "{task_title}".
            """,
            input_variables=["task_title", "design_doc"],
        )
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", convert_system_message_to_human=True)
        chain = prompt | llm
        implementation_plan = chain.invoke({
            "task_title": current_item['title'],
            "design_doc": state['design_doc']
        }).content
        
        task_id = current_item.get('id', 'unnamed_task')
        filename = f"{task_id.replace(' ', '_').lower()}.md"
        filepath = f"/workspaces/agentic-dev-platform/workspace/{filename}"
        
        with open(filepath, "w") as f:
            f.write(f"# Implementation Plan for: {current_item['title']}\n\n")
            f.write(implementation_plan)
        
        change_record = f"CREATED: {filepath}"
        log.append(f"[Coder] Success: Generated and wrote implementation plan to {filepath}")

        # --- Step 2: NEW - Execute a shell command using the tool ---
        log.append(f"[Coder] Info: Now executing a verification command in the workspace.")
        command_to_run = "ls -l /opt/workspace_base"
        exit_code, output = run_shell_command(command_to_run)
        
        if exit_code == 0:
            log.append(f"[Coder] Success: Shell command '{command_to_run}' executed successfully.")
            log.append(f"[Coder] Shell Output:\n---\n{output}\n---")
        else:
            log.append(f"[Coder] Failure: Shell command '{command_to_run}' failed with exit code {exit_code}.")
            log.append(f"[Coder] Shell Error Output:\n---\n{output}\n---")
            return {**state, "error": f"Shell command failed: {output}", "run_log": log}


        # --- Step 3: Update Roadmap (as before) ---
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
        
        if not task_found_and_updated:
            for task in updated_roadmap:
                 if isinstance(task, dict) and task.get("id") == current_item.get("id"):
                    task["status"] = "done"
                    break

        log.append(f"[Coder] Info: Marked task '{task_id}' as 'done' in the roadmap.")
        return {**state, "repo_changes": [change_record], "roadmap": updated_roadmap, "run_log": log}

    except Exception as e:
        log.append(f"[Coder] Failure: Coder agent failed for task {current_item.get('id', 'unknown')}. Details: {e}")
        return {**state, "error": f"Coder agent failed: {e}", "run_log": log}


def tester(state: State) -> State:
    """
    Agent that simulates running tests on the changes made by the Coder.
    """
    print("---EXECUTING TESTER---")
    log = state.get("run_log", [])
    repo_changes = state.get("repo_changes")
    if repo_changes:
        log.append(f"[Tester] Success: Simulated tests PASSED for the following changes: {', '.join(repo_changes)}")
        return {**state, "repo_changes": [], "run_log": log}
    else:
        log.append("[Tester] Info: No file changes were detected to test.")
        return {**state, "run_log": log}


def cicd(state: State) -> State:
    """
    Agent that simulates a CI/CD pipeline.
    """
    print("---EXECUTING CICD---")
    log = state.get("run_log", [])
    current_item = state.get("current_item")
    if current_item and current_item.get("status") == "done":
         log.append(f"[CICD] Success: Simulated build and deployment for task '{current_item.get('title', 'N/A')}'. Artifacts are ready.")
    else:
        log.append(f"[CICD] Info: No completed task to process for deployment.")
    return {**state, "run_log": log}


def log_analyst(state: State) -> State:
    """
    Agent that is triggered on failure to analyze logs.
    """
    print("---EXECUTING LOG ANALYST---")
    log = state.get("run_log", [])
    error_message = state.get('error', 'Unknown error')
    log.append(f"[LogAnalyst] Info: An error was detected. Analyzing details: {error_message}")
    return {**state, "run_log": log}


# === 3. Conditional Edges ===
def should_continue(state: State) -> Literal["Architect", "__end__"]:
    """
    Determines if the main workflow should continue to the Architect or end.
    """
    if state.get("current_item") is not None:
        return "Architect"
    else:
        return "__end__"

def check_for_errors(state: State) -> Literal["LogAnalyst", "Planner", "Architect", "Tester", "Coder", "CICD"]:
    """
    Routes the workflow to the LogAnalyst if an error is present.
    """
    if state.get("error"):
        return "LogAnalyst"
    return "CICD" 


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
    print("\n---Invoking agent workflow---")
    
    initial_state = {"run_log": [], "metadata": {"project": "agentic-dev-platform"}}
    config = {"configurable": {"thread_id": "proj-thread-1"}, "recursion_limit": 250}

    final_state = graph.invoke(initial_state, config=config)
    
    print("\n---Workflow Complete---")
    
    log_content = "\n".join(final_state.get("run_log", ["No log entries found."]))
    with open("/workspaces/agentic-dev-platform/run_log.txt", "w") as f:
        f.write("AGENTIC DEV PLATFORM - RUN LOG\n")
        f.write("=================================\n\n")
        f.write(log_content)
        
    print("Final run log has been written to run_log.txt")

