"""
PubMed Tool
"""

import os

from Bio import Entrez
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

Entrez.email = os.getenv("EMAIL") or "anonymous@example.com"


@tool
def pubmed_search(query: str) -> str:
    """
    Search PubMed and return top 5 paper titles.
    """
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=5,
    )
    record = Entrez.read(handle)
    ids = record["IdList"]

    if not ids:
        return "No PubMed papers found."

    handle = Entrez.esummary(
        db="pubmed",
        id=",".join(ids),
    )
    summaries = Entrez.read(handle)

    papers = [f"- {paper['Title']}" for paper in summaries]
    return "\n".join(papers)
