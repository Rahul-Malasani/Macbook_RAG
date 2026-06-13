# clirag/retriever.py
# Turns a query string into ranked context chunks. Thin by design — this is the seam
# where retrieval experiments plug in (reranking, hybrid BM25+vector, MMR, relevance
# threshold). Baseline: plain top-k cosine similarity, no filtering.
from clirag.config import TOP_K
from clirag.vectorstore import query as vs_query


class Retriever:
    def __init__(self, embedder, collection, top_k: int = TOP_K):
        self.embedder = embedder
        self.collection = collection
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        """Embed the query, fetch the top_k nearest chunks, and enrich each with a
        1-distance cosine similarity and its rank (0 = closest)."""
        query_embedding = self.embedder.embed_query(query)
        hits = vs_query(self.collection, query_embedding, top_k=self.top_k)
        for rank, hit in enumerate(hits):
            hit["similarity"] = 1.0 - hit["distance"]
            hit["rank"] = rank
        return hits
