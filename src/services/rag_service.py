"""
RAG Service
"""

from src.rag.agents.graph import graph


class RagService:

    def ask(
        self,
        question,
        prediction,
        confidence,
    ):

        state = graph.invoke(

            {
                "question": question,
                "prediction": prediction,
                "confidence": confidence,
            }

        )

        return state