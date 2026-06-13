# clirag/ingest.py
# Orchestrates the ingest pipeline: load -> chunk -> embed -> store. No logic of its own.
# Run:  python -m clirag.ingest
import json

from clirag.chunker import chunk_documents, save_chunks
from clirag.config import CORPUS_MANIFEST
from clirag.embedder import get_embedder
from clirag.loader import load_documents
from clirag.vectorstore import count, get_collection, reset, upsert_chunks


def _corpus_version():
    try:
        with open(CORPUS_MANIFEST) as f:
            return json.load(f)["corpus_version"]
    except (FileNotFoundError, KeyError):
        return None


def ingest(rebuild: bool = True) -> None:
    print("=== ingest ===")
    corpus_version = _corpus_version()
    docs = load_documents()
    chunks = chunk_documents(docs, corpus_version=corpus_version)
    save_chunks(chunks)
    embedder = get_embedder()
    vectors = embedder.embed_documents([c.text for c in chunks])
    if rebuild:
        reset()                          # drop any previous collection for a clean build
    collection = get_collection()
    upsert_chunks(collection, chunks, vectors)
    print(f"=== done: {count(collection)} chunks stored (corpus_version={corpus_version}) ===")


if __name__ == "__main__":
    ingest()
