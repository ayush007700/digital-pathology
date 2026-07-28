from langgraph.graph import END, StateGraph

from src.rag.agents.general_agent import general_agent
from src.rag.agents.pathology_agent import pathology_agent
from src.rag.agents.retriever_agent import retriever_agent
from src.rag.agents.state import GraphState
from src.rag.agents.supervisor import supervisor
from src.rag.agents.tool_agent import tool_agent
from src.rag.memory import memory


def router(state: GraphState) -> str:
    """Extracts the next node destination decided by the supervisor."""
    return state["next"]


builder = StateGraph(GraphState)

builder.add_node("supervisor", supervisor)
builder.add_node("retriever", retriever_agent)
builder.add_node("pathology", pathology_agent)
builder.add_node("general", general_agent)
builder.add_node("tool", tool_agent)

builder.set_entry_point("supervisor")

builder.add_conditional_edges(
    "supervisor",
    router,
    {
        "retriever": "retriever",
        "general": "general",
        "tool": "tool",
    },
)

builder.add_edge("retriever", "pathology")
builder.add_edge("pathology", END)
builder.add_edge("general", END)
builder.add_edge("tool", END)

graph = builder.compile(checkpointer=memory)
