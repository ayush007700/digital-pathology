from src.rag.retriever import retrieve


def retriever_agent(state):

    docs = retrieve(state["question"])

    context = "\n\n".join(

        d.page_content

        for d in docs

    )

    state["context"] = context

    return state