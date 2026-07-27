from src.rag.vectorstore import db

retriever = db.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5, "k": 4}
)


def retrieve(query):
    return retriever.invoke(query)
