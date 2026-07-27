from src.rag.tools.pubmed_tool import pubmed_search

from src.rag.tools.calculator_tool import calculator


TOOLS = [

    pubmed_search,

    calculator,

]