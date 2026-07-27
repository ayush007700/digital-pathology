"""
General Agent
"""

from src.rag.llm import llm


def general_agent(state):

    response = llm.invoke(

        state["question"]

    )

    state["answer"] = response.content

    return state