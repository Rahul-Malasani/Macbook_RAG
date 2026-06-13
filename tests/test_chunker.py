import json

import pytest

from clirag.chunker import chunk_documents, save_chunks, split_text
from clirag.models import Document


def _doc(text, tool="x"):
    return Document(text=text, metadata={"source": f"data/raw/man_{tool}.txt", "tool": tool})


def test_overlap_ge_size_raises():
    with pytest.raises(ValueError):
        chunk_documents([_doc("a b c d")], chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        split_text("a b c d", 50, 80)


def test_nonpositive_size_raises():
    with pytest.raises(ValueError):
        chunk_documents([_doc("a b c")], chunk_size=0, overlap=0)


def test_empty_document_yields_no_chunks():
    assert chunk_documents([_doc("   \n  ")]) == []


def test_deterministic_and_unique_ids():
    docs = [_doc("word " * 500, "a"), _doc("line\n" * 400, "b")]
    a = chunk_documents(docs, chunk_size=200, overlap=40)
    b = chunk_documents(docs, chunk_size=200, overlap=40)
    assert len(a) > 1
    assert [c.id for c in a] == [c.id for c in b]        # deterministic across runs
    assert len({c.id for c in a}) == len(a)              # unique within a run


def test_chunk_size_respected():
    text = " ".join(f"w{i}" for i in range(1000))        # small, splittable tokens
    parts = split_text(text, chunk_size=200, overlap=40)
    assert parts and all(len(p) <= 200 for p in parts)


def test_overlap_present_between_adjacent_chunks():
    text = " ".join(f"w{i:03d}" for i in range(200))
    parts = split_text(text, chunk_size=120, overlap=40)
    assert len(parts) >= 2
    # the tail of one chunk should reappear at the head of the next (word-level overlap)
    assert set(parts[0].split()[-3:]) & set(parts[1].split()[:6])


def test_metadata_and_index():
    chunks = chunk_documents([_doc("alpha " * 300, "grep")],
                             chunk_size=150, overlap=30, corpus_version="deadbeef")
    assert all(c.metadata["tool"] == "grep" for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["corpus_version"] == "deadbeef" for c in chunks)
    assert all(c.metadata["char_len"] == len(c.text) for c in chunks)


def test_save_chunks_roundtrip(tmp_path):
    chunks = chunk_documents([_doc("hello world " * 50, "ls")], chunk_size=100, overlap=20)
    path = tmp_path / "chunks.json"
    save_chunks(chunks, str(path))
    data = json.loads(path.read_text())
    assert len(data) == len(chunks)
    assert data[0]["id"] == chunks[0].id
    assert data[0]["metadata"]["tool"] == "ls"


def test_real_corpus_chunks():
    from clirag.loader import load_documents

    chunks = chunk_documents(load_documents())
    assert len(chunks) > 200                              # 128 pages -> many chunks
    assert len({c.id for c in chunks}) == len(chunks)     # ids unique across the corpus
