import pytest

from clirag.models import Chunk
from clirag.vectorstore import count, get_collection, query, upsert_chunks


def _fixture():
    # 3 chunks across 2 sources; axis-aligned vectors so nearest-neighbour is deterministic.
    chunks = [
        Chunk("a1", "copy files with cp", {"source": "man_cp.txt", "tool": "cp", "chunk_index": 0}),
        Chunk("g1", "search text with grep", {"source": "man_grep.txt", "tool": "grep", "chunk_index": 0}),
        Chunk("g2", "grep options", {"source": "man_grep.txt", "tool": "grep", "chunk_index": 1}),
    ]
    vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return chunks, vecs


def test_upsert_and_count(tmp_path):
    col = get_collection(path=str(tmp_path))
    chunks, vecs = _fixture()
    upsert_chunks(col, chunks, vecs)
    assert count(col) == 3


def test_query_returns_nearest(tmp_path):
    col = get_collection(path=str(tmp_path))
    upsert_chunks(col, *_fixture())
    hit = query(col, [1.0, 0.0, 0.0], top_k=1)[0]
    assert hit["id"] == "a1"
    assert hit["metadata"]["tool"] == "cp"
    assert 0.0 <= hit["distance"] <= 2.0          # cosine distance range


def test_upsert_is_idempotent(tmp_path):
    col = get_collection(path=str(tmp_path))
    chunks, vecs = _fixture()
    upsert_chunks(col, chunks, vecs)
    upsert_chunks(col, chunks, vecs)              # same content again
    assert count(col) == 3                        # not 6


def test_delete_by_source_on_reingest(tmp_path):
    col = get_collection(path=str(tmp_path))
    upsert_chunks(col, *_fixture())
    # re-ingest the grep page with a single, different chunk
    new = [Chunk("g_new", "grep rewritten", {"source": "man_grep.txt", "tool": "grep", "chunk_index": 0})]
    upsert_chunks(col, new, [[0.0, 0.5, 0.5]])
    assert set(col.get()["ids"]) == {"a1", "g_new"}   # cp kept, old grep chunks gone


def test_length_mismatch_raises(tmp_path):
    col = get_collection(path=str(tmp_path))
    with pytest.raises(ValueError):
        upsert_chunks(col, [Chunk("x", "t", {"source": "s"})], [])
