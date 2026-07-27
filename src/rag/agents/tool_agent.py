"""
Tool Agent — invoke tools, then ask the LLM for a final answer.
"""

from langchain_core.messages import HumanMessage, ToolMessage

from src.rag.llm import llm_with_tools
from src.rag.tools.tools import TOOLS

TOOL_MAP = {t.name: t for t in TOOLS}


def tool_agent(state):
    question = state["question"]
    messages = [HumanMessage(content=question)]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # Model asked for tools: run them, then get a final natural-language answer
    if response.tool_calls:
        for call in response.tool_calls:
            tool = TOOL_MAP.get(call["name"])
            if tool is None:
                result = f"Unknown tool: {call['name']}"
            else:
                result = tool.invoke(call["args"])

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"],
                )
            )

        final = llm_with_tools.invoke(messages)
        answer = final.content
        tool_calls = response.tool_calls
    else:
        answer = response.content
        tool_calls = []

    state["answer"] = answer or "(no answer produced)"
    state["tool_calls"] = tool_calls
    return state
