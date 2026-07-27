from src.rag.tools.pubmed_tool import pubmed_search

print(

    pubmed_search.invoke(

        {

            "query":"HER2 breast cancer"

        }

    )

)