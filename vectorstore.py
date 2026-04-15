# vectorstore.py
# Responsibility: all ChromaDB interactions — saving chunks and loading for querying.
# Two functions: one for ingest path (save), one for retrieval path (load).

from langchain_chroma import Chroma
from config import CHROMA_DIR


def save_to_vectorstore(chunks, embedder):
    """
    Embeds chunks and stores them in ChromaDB.
    Called once per ingest run.
    ChromaDB persists to CHROMA_DIR on disk — no need to re-embed on next query.
    """
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=CHROMA_DIR
    )
    print(f"[vectorstore] Stored {len(chunks)} chunks in ChromaDB at '{CHROMA_DIR}'")
    return vectorstore


def load_vectorstore(embedder):
    """
    Loads existing ChromaDB from disk for querying.
    Uses the same embedder to ensure query vectors are in the same space as stored vectors.
    """
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedder
    )
    print(f"[vectorstore] Loaded ChromaDB from '{CHROMA_DIR}'")
    return vectorstore
