"""
Simple RAG QA Chain
"""

from src.rag.llm import llm
from src.rag.prompts import RAG_PROMPT
from src.rag.retriever import retrieve


def ask(question: str):

    docs = retrieve(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = RAG_PROMPT.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": [
            d.metadata.get("source", "Unknown")
            for d in docs
        ],
    }