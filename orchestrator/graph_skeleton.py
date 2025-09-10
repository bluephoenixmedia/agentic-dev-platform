from typing import TypedDict, List, Dict, Optional
import sys
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

class State(TypedDict, total=False):
    design_doc: str
    roadmap: List[dict]
    current_item: Optional[dict]
    repo_changes: List[str]
    run_status: str
    logs_tail: str
    approvals: Dict[str, bool]
    metadata: Dict[str, str]
    # A new field to track errors for conditional routing
    error: Optional[str]

def doc_agent(state: State) -> State:
    print("---EXECUTING DOC AGENT---")
    return state

def planner(state: State) -> State:
    print("---EXECUTING PLANNER---")
    return state

def architect(state: State) -> State:
    print("---EXECUTING ARCHITECT---")
    return state

def coder(state: State) -> State:
    print("---EXECUTING CODER---")
    # In a real implementation, this would be set if tests fail
    state["error"] = None 
    return state

def tester(state: State) -> State:
    print("---EXECUTING TESTER---")
    # In a real implementation, this would be set if tests fail
    state["error"] = None
    return state

def cicd(state: State) -> State:
    print("---EXECUTING CICD---")
    return state

def log_analyst(state: State) -> State:
    print("---EXECUTING LOG ANALYST---")
    return state

# This function is our conditional router
def should_continue(state: State) -> str:
    """Determines whether to continue to the next step or go to the error handler."""
    if state.get("error"):
        return "error"
    return "continue"

builder = StateGraph(State)
builder.add_node("DocAgent", doc_agent)
builder.add_node("Planner", planner)
builder.add_node("Architect", architect)
builder.add_node("Coder", coder)
builder.add_node("Tester", tester)
builder.add_node("CICD", cicd)
builder.add_node("LogAnalyst", log_analyst)

builder.set_entry_point("DocAgent")

# Main workflow path
builder.add_edge("DocAgent", "Planner")
builder.add_edge("Planner", "Architect")
builder.add_edge("Architect", "Coder")

# Conditional routing for Coder and Tester
builder.add_conditional_edges(
    "Coder",
    should_continue,
    {
        "continue": "Tester",
        "error": "LogAnalyst",
    },
)
builder.add_conditional_edges(
    "Tester",
    should_continue,
    {
        "continue": "CICD",
        "error": "LogAnalyst",
    },
)

builder.add_edge("LogAnalyst", "Coder") # Error-handling loop
builder.add_edge("CICD", END)

graph = builder.compile(checkpointer=SqliteSaver.from_conn_string(":memory:"))

print("Graph compiled successfully.")

# --- Main execution block ---
if __name__ == "__main__":
    print("\n---Invoking agent workflow---")
    initial_state = {"metadata": {"project": "agentic-dev-platform"}}
    config = {"configurable": {"thread_id": "main-thread"}}
    
    # Stream the events to see the flow of execution
    for event in graph.stream(initial_state, config=config):
        for key, value in event.items():
            print(f"Event: {key}")
            print("---")
            print(value)
        print("\n" + "="*30 + "\n")

