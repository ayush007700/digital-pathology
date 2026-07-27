from src.rag.agents.graph import graph

config = {
    "configurable": {
        "thread_id": "demo-session"
    }
}

for event in graph.stream(
    {
        "question": "What are the pathological features of HER2 breast cancer?"
    },
    config=config,
):
    print("=" * 80)
    print(event)