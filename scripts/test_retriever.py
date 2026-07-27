from src.rag.retriever import retrieve

docs = retrieve("What is breast cancer?")

for d in docs:

    print("=" * 50)

    print(d.page_content[:300])
