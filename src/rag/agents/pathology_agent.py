from src.rag.llm import llm

from src.rag.prompts import RAG_PROMPT


def pathology_agent(state):

    prompt = RAG_PROMPT.invoke(

        {

            "context": state["context"],

            "question": state["question"],

        }

    )

    response = llm.invoke(prompt)

    state["answer"] = response.content

    return state