# ingest.py
# Responsibility: orchestrate the ingest pipeline.
# Contains NO logic of its own — just calls the right modules in order.
# Pipeline: load → chunk → embed → store

from dotenv import load_dotenv
from loader import load_documents
from chunker import chunk_documents
from embedder import get_embedder
from vectorstore import save_to_vectorstore

load_dotenv()


def ingest():
    print("=== Starting Ingest Pipeline ===\n")

    # Step 1: Load raw .txt files from data/raw/
    documents = load_documents()

    # Step 2: Split into chunks + save to data/processed/chunks.json
    chunks = chunk_documents(documents)

    # Step 3: Get embedding model
    embedder = get_embedder()

    # Step 4: Embed chunks and store in ChromaDB
    save_to_vectorstore(chunks, embedder)

    print("\n=== Ingest Complete ===")


if __name__ == "__main__":
    ingest()
