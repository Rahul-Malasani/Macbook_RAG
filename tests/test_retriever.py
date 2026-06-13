from clirag.models import Chunk
from clirag.retriever import Retriever
from clirag.vectorstore import get_collection, upsert_chunks


class _FakeEmbedder:
    """Returns a fixed query vector, so ranking is determined by the stored vectors."""

    def __init__(self, vec):
        self.vec = vec

    def embed_query(self, text):
        return self.vec


def _collection(tmp_path):
    col = get_collection(path=str(tmp_path))
    chunks = [
        Chunk("a1", "copy files", {"source": "man_cp.txt", "tool": "cp", "chunk_index": 0}),
        Chunk("g1", "search text", {"source": "man_grep.txt", "tool": "grep", "chunk_index": 0}),
        Chunk("s1", "remote login", {"source": "man_ssh.txt", "tool": "ssh", "chunk_index": 0}),
    ]
    vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    upsert_chunks(col, chunks, vecs)
    return col


def test_retrieve_ranks_and_similarity(tmp_path):
    r = Retriever(_FakeEmbedder([1.0, 0.0, 0.0]), _collection(tmp_path), top_k=3)
    hits = r.retrieve("anything")
    assert len(hits) == 3
    assert hits[0]["id"] == "a1"                              # nearest first
    assert [h["rank"] for h in hits] == [0, 1, 2]            # ranks in order
    for h in hits:
        assert abs(h["similarity"] - (1.0 - h["distance"])) < 1e-9
    sims = [h["similarity"] for h in hits]
    assert sims == sorted(sims, reverse=True)                # closest = highest similarity


def test_top_k_respected(tmp_path):
    r = Retriever(_FakeEmbedder([0.0, 1.0, 0.0]), _collection(tmp_path), top_k=1)
    hits = r.retrieve("q")
    assert len(hits) == 1
    assert hits[0]["id"] == "g1"                              # query nearest the grep vector
