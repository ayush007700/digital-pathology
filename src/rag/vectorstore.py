from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag.settings import CHROMA_DIR

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding,
)
