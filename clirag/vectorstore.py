# clirag/vectorstore.py
# Stage 4 (ingest) + the retrieval read side: persist chunk vectors to ChromaDB and
# query them. Raw `chromadb` client — no LangChain.
#
# We always pass explicit embeddings / query_embeddings, so Chroma never falls back to
# its built-in embedding function (which would download an ONNX model).
import chromadb

from clirag.config import CHROMA_DIR, COLLECTION_NAME, DISTANCE, TOP_K
from clirag.models import Chunk


def get_collection(path=CHROMA_DIR, name=COLLECTION_NAME, distance=DISTANCE):
    """Persistent collection configured with an explicit distance metric (cosine)."""
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": distance})


def upsert_chunks(collection, chunks: list[Chunk], embeddings: list[list[float]],
                  batch_size: int = 5000) -> None:
    """Idempotently write chunks: delete any existing vectors for the same sources, then
    add. Re-ingesting a page replaces its chunks — no duplicates, no orphans within a source."""
    if not chunks:
        return
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
        )
    sources = sorted({c.metadata.get("source", "") for c in chunks})
    collection.delete(where=_where_sources(sources))
    # Chroma caps a single add (~5461 rows); batch to stay under it.
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        collection.add(
            ids=[c.id for c in batch],
            embeddings=embeddings[start:start + batch_size],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
        )


def query(collection, query_embedding: list[float], top_k: int = TOP_K) -> list[dict]:
    """Return the top_k nearest chunks as [{id, text, metadata, distance}] (raw cosine
    distance; the retriever converts to a 1-distance similarity for display)."""
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i],
        }
        for i in range(len(res["ids"][0]))
    ]


def count(collection) -> int:
    return collection.count()


def reset(path=CHROMA_DIR, name=COLLECTION_NAME) -> None:
    """Drop the collection for a clean full rebuild."""
    client = chromadb.PersistentClient(path=path)
    try:
        client.delete_collection(name=name)
    except Exception:
        pass  # collection didn't exist — nothing to drop


def _where_sources(sources):
    # Chroma wants $in for multiple values; single-key equality for one source.
    return {"source": sources[0]} if len(sources) == 1 else {"source": {"$in": sources}}
