# embedder.py
# Responsibility: configure and return the embedding model.
# Kept separate so swapping embedding models is a one-line change in config.py.
# The actual embedding computation happens inside the vector store — this file
# just sets up the model object that the vector store will call.

from langchain_ollama import OllamaEmbeddings
from config import EMBEDDING_MODEL


def get_embedder():
    """
    Returns a configured OllamaEmbeddings instance.
    IMPORTANT: the same embedder must be used at ingest time AND query time.
    If you change EMBEDDING_MODEL in config.py, delete chroma_db/ and re-ingest.
    """
    embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)
    print(f"[embedder] Using embedding model: '{EMBEDDING_MODEL}'")
    return embedder
