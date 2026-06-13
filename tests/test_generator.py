import pytest

import clirag.generator as gen_module
from clirag.generator import IDK, Generator


class _FakeResp:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()


def test_empty_hits_short_circuits(monkeypatch):
    called = []
    monkeypatch.setattr(gen_module.ollama, "chat",
                        lambda **k: called.append(k) or _FakeResp("x"))
    assert Generator().generate("q", []) == IDK
    assert called == []                              # no LLM call when nothing retrieved


def test_prompt_contains_context_question_and_grounding(monkeypatch):
    captured = {}

    def fake_chat(model, messages, options):
        captured["messages"] = messages
        return _FakeResp("  the answer  ")

    monkeypatch.setattr(gen_module.ollama, "chat", fake_chat)
    out = Generator().generate("how to copy?", [{"text": "cp copies files", "metadata": {"tool": "cp"}}])
    assert out == "the answer"                        # stripped
    joined = " ".join(m["content"] for m in captured["messages"])
    assert "cp copies files" in joined               # context included
    assert "how to copy?" in joined                  # question included
    assert IDK in joined                             # grounding instruction present


def test_connection_error_is_wrapped(monkeypatch):
    def boom(**k):
        raise ConnectionError("refused")

    monkeypatch.setattr(gen_module.ollama, "chat", boom)
    with pytest.raises(RuntimeError, match="ollama serve"):
        Generator().generate("q", [{"text": "t", "metadata": {"tool": "x"}}])
