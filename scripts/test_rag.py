from src.rag.qa_chain import ask

response = ask(

    "What are the histological features of breast cancer?"

)

print()

print("=" * 80)

print(response["answer"])

print()

print("=" * 80)

print("Sources")

for source in response["sources"]:

    print(source)