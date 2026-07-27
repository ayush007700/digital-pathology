from src.rag.agents.graph import graph

response = graph.invoke(

    {

        "question":

        "What are the pathological features of breast cancer?"

    }

)

print()

print(response["answer"])