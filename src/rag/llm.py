"""
OpenAI LLM

Author: Ayush Raj
"""

import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from src.rag.tools.tools import TOOLS

load_dotenv()

llm = ChatOpenAI(

    model=os.getenv("OPENAI_MODEL"),

    temperature=0,

)

llm_with_tools = llm.bind_tools(TOOLS)