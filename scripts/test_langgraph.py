from src.rag.agents.graph import graph

questions = [

    "What is breast cancer?",

    "What is 125*32?",

    "Search PubMed for HER2 breast cancer.",

]
for q in questions:

    print("=" * 80)

    print("QUESTION:", q)

    result = graph.invoke(

        {

            "question": q,

        }

    )

    print()

    print(result["answer"])