# clirag/embedder.py
# Stage 3 (and the query side): turn text into vectors with Ollama's nomic-embed-text.
# Raw `ollama` client — no LangChain.
#
# Baseline uses NO task prefixes. nomic-embed-text is trained with "search_document:" /
# "search_query:" prefixes that usually help retrieval; adding them is a measured
# experiment (see DECISIONS.md). embed_documents / embed_query are split now so that's a
# one-line change later.
import ollama

from clirag.config import EMBEDDING_DIM, EMBEDDING_MODEL


class Embedder:
    """Wraps one Ollama embedding model. The SAME instance/model must be used at ingest
    and query time, or stored vectors and query vectors live in different spaces."""

    def __init__(self, model: str = EMBEDDING_MODEL, dim: int = EMBEDDING_DIM):
        self.model = model
        self.dim = dim

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            vecs = self._embed(batch)
            if start == 0 and vecs and len(vecs[0]) != self.dim:
                raise ValueError(
                    f"expected {self.dim}-dim vectors from '{self.model}', got {len(vecs[0])}"
                )
            vectors.extend(vecs)
            done = min(start + batch_size, len(texts))
            print(f"\r[embedder] {done}/{len(texts)}", end="", flush=True)
        print()
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _embed(self, batch: list[str]) -> list[list[float]]:
        try:
            resp = ollama.embed(model=self.model, input=batch)
        except Exception as e:  # broad on purpose; step 6 adds timeouts/retries
            raise RuntimeError(
                f"Ollama embed failed for '{self.model}'. Is `ollama serve` running "
                f"and `{self.model}` pulled? ({e})"
            ) from e
        vectors = getattr(resp, "embeddings", None)
        if vectors is None:
            vectors = resp["embeddings"]
        return [list(v) for v in vectors]


def get_embedder() -> Embedder:
    print(f"[embedder] model={EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    return Embedder()
