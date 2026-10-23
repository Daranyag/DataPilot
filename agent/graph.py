from langgraph.graph import StateGraph, END
from .nodes import AgentState, generate_code_node, execute_code_node, explain_result_node

def should_retry(state: AgentState) -> str:
    """Routing function to decide if we should retry code generation or move to explanation."""
    # Skip explanation if it's already generated (e.g. FIT_CHECK)
    if state.get("final_explanation"):
        return "end"
        
    if state["execution_error"]:
        if state["retries"] < state["max_retries"]:
            return "generate_code"
        else:
            return "end" # Max retries reached, exit with error
    return "explain_result" # Success, explain result

def build_graph() -> StateGraph:
    """Builds and returns the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("generate_code", generate_code_node)
    workflow.add_node("execute_code", execute_code_node)
    workflow.add_node("explain_result", explain_result_node)
    
    # Add edges
    workflow.set_entry_point("generate_code")
    workflow.add_edge("generate_code", "execute_code")
    workflow.add_conditional_edges(
        "execute_code",
        should_retry,
        {
            "generate_code": "generate_code",
            "explain_result": "explain_result",
            "end": END
        }
    )
    workflow.add_edge("explain_result", END)
    
    return workflow.compile()
