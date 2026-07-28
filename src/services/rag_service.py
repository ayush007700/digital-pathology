"""
RAG Service
"""

import uuid

from src.rag.agents.graph import graph


class RagService:

    def ask(
        self,
        question,
        prediction,
        confidence,
    ):
        return graph.invoke(
            {
                "question": question,
                "prediction": prediction,
                "confidence": confidence,
            },
            config={
                "configurable": {
                    "thread_id": str(uuid.uuid4()),
                }
            },
        )
