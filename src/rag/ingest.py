from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.settings import KNOWLEDGE_DIR
from src.rag.vectorstore import db


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

pdfs = list(KNOWLEDGE_DIR.rglob("*.pdf"))
if not pdfs:
    raise SystemExit(
        f"No PDFs found in {KNOWLEDGE_DIR.resolve()}. "
        "Add PDF files there, then re-run: python -m src.rag.ingest"
    )

docs = []
for pdf in pdfs:
    loader = PyPDFLoader(str(pdf))
    docs.extend(loader.load())

chunks = splitter.split_documents(docs)
if not chunks:
    raise SystemExit("PDFs loaded but produced 0 chunks (empty or unreadable content).")

db.add_documents(chunks)
print(f"Ingested {len(chunks)} chunks from {len(pdfs)} PDF(s)")
