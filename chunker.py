# chunker.py
# Responsibility: split Document objects into chunks, save to JSON as checkpoint.
# This is the stage your earlier ingest.py skipped — processed chunks are now
# saved to data/processed/ so you can inspect, audit, and re-embed without
# reloading raw files.

import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, PROCESSED_DATA_DIR


def chunk_documents(documents):
    """
    Splits a list of Documents into smaller chunks using RecursiveCharacterTextSplitter.
    Saves chunks to data/processed/chunks.json as an auditable checkpoint.
    Returns the list of chunk Documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"[chunker] Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # Save to JSON so you can inspect exactly what the vector store will receive
    _save_chunks(chunks)
    return chunks


def _save_chunks(chunks):
    """
    Saves chunks to data/processed/chunks.json.
    Each entry has the chunk text and its metadata (source filename etc).
    This is your human-readable audit trail of what got embedded.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, "chunks.json")

    data = [
        {
            "index": i,
            "text": chunk.page_content,
            "metadata": chunk.metadata
        }
        for i, chunk in enumerate(chunks)
    ]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[chunker] Saved {len(chunks)} chunks to '{output_path}'")
