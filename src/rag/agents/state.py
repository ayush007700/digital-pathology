"""
Shared LangGraph State
"""

from typing import TypedDict


class GraphState(TypedDict):

    question: str

    context: str

    answer: str

    next: str

    prediction: str

    confidence: float

    report_path: str

    tool_calls: list