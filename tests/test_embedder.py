import pytest

import clirag.embedder as embedder_module
from clirag.embedder import Embedder


class _FakeResp:
    def __init__(self, embeddings):
        self.embeddings = embeddings


def _stub(monkeypatch, dim=768):
    """Replace ollama.embed with a recorder that returns dim-sized zero vectors."""
    calls = []

    def fake_embed(model, input):
        calls.append(list(input))
        return _FakeResp([[0.0] * dim for _ in input])

    monkeypatch.setattr(embedder_module.ollama, "embed", fake_embed)
    return calls


def test_embed_documents_batches_correctly(monkeypatch):
    calls = _stub(monkeypatch)
    vecs = Embedder().embed_documents([f"t{i}" for i in range(130)], batch_size=64)
    assert len(vecs) == 130
    assert len(vecs[0]) == 768
    assert [len(c) for c in calls] == [64, 64, 2]   # batched as 64 + 64 + 2


def test_embed_query_returns_single_vector(monkeypatch):
    _stub(monkeypatch)
    v = Embedder().embed_query("how do I copy a file?")
    assert len(v) == 768


def test_empty_input_returns_empty(monkeypatch):
    _stub(monkeypatch)
    assert Embedder().embed_documents([]) == []


def test_dimension_mismatch_raises(monkeypatch):
    _stub(monkeypatch, dim=100)        # model returns wrong size
    with pytest.raises(ValueError):
        Embedder().embed_documents(["x"])


def test_connection_failure_is_wrapped(monkeypatch):
    def boom(model, input):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(embedder_module.ollama, "embed", boom)
    with pytest.raises(RuntimeError, match="ollama serve"):
        Embedder().embed_query("x")
