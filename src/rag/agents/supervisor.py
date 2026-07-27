"""
Supervisor Agent
"""

from src.rag.llm import llm

VALID_ROUTES = {"retriever", "general", "tool"}

ROUTER_PROMPT = """
You are an AI supervisor.

Choose ONLY ONE route.

Available routes:
- retriever: pathology / medical questions that need document context
- general: general knowledge questions
- tool: questions that need calculation or PubMed search

Question:
{question}

Return ONLY the route name (retriever, general, or tool).
"""


def supervisor(state):
    response = llm.invoke(
        ROUTER_PROMPT.format(question=state["question"])
    )

    route = response.content.strip().lower()
    # Keep only the first token in case the model adds extra text
    route = route.replace("`", "").split()[0] if route else "general"

    # Map legacy/alias names onto real graph nodes
    aliases = {
        "pathology": "retriever",
        "rag": "retriever",
        "tools": "tool",
        "calculator": "tool",
        "pubmed": "tool",
    }
    route = aliases.get(route, route)

    if route not in VALID_ROUTES:
        route = "general"

    state["next"] = route
    return state
