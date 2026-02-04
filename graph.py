from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from schema import AgentState
from nodes import refine_node, gather_info_node, generator_node, updater_node, ts_generator_node

def should_continue_gather(state: AgentState):
    """
    Determines if we should continue gathering or move to generation.
    """
    if not state.get("missing_info"):
        return "generator_node"
    return END

def route_start(state: AgentState):
    """
    Decides where to start based on existing state and user intent.
    """
    last_msg = state['messages'][-1]['content'].upper() if state['messages'] else ""
    
    # If a document already exists, determine if user wants an UPDATE or a TS
    if state.get("draft_html"):
         if "TECHNICAL SPECIFICATION" in last_msg or " TSD " in last_msg or " TS " in last_msg:
              return "ts_generator_node"
         return "updater_node"
    
    if state.get("user_goal"):
        return "gather_info_node"
    return "refine_node"

workflow = StateGraph(AgentState)

workflow.add_node("refine_node", refine_node)
workflow.add_node("gather_info_node", gather_info_node)
workflow.add_node("generator_node", generator_node)
workflow.add_node("updater_node", updater_node)
workflow.add_node("ts_generator_node", ts_generator_node)

workflow.set_conditional_entry_point(
    route_start,
    {
        "refine_node": "refine_node",
        "gather_info_node": "gather_info_node",
        "updater_node": "updater_node",
        "ts_generator_node": "ts_generator_node"
    }
)

workflow.add_edge("refine_node", "gather_info_node")
workflow.add_edge("updater_node", "generator_node") # Re-gen BRD after update
workflow.add_edge("ts_generator_node", END)

workflow.add_conditional_edges(
    "gather_info_node",
    should_continue_gather,
    {
        "generator_node": "generator_node",
        END: END
    }
)

workflow.add_edge("generator_node", END)

# Add memory for session persistence (thread-based)
memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)
