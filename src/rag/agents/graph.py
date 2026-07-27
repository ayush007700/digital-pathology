from langgraph.graph import StateGraph

from langgraph.graph import END

from src.rag.agents.state import GraphState

from src.rag.agents.supervisor import supervisor

from src.rag.agents.retriever_agent import retriever_agent

from src.rag.agents.pathology_agent import pathology_agent

from src.rag.agents.report_agent import report_agent


builder = StateGraph(GraphState)

builder.add_node(

    "supervisor",

    supervisor,

)

builder.add_node(

    "retriever",

    retriever_agent,

)

builder.add_node(

    "pathology",

    pathology_agent,

)

builder.add_node(

    "report",

    report_agent,

)

builder.set_entry_point(

    "supervisor"

)

builder.add_edge(

    "supervisor",

    "retriever",

)

builder.add_edge(

    "retriever",

    "pathology",

)

builder.add_edge(

    "pathology",

    "report",

)

builder.add_edge(

    "report",

    END,

)

graph = builder.compile()