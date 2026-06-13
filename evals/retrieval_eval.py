# evals/retrieval_eval.py
# Retrieval metrics over the golden set: hit@k and MRR at the tool level (one man page =
# one tool in this corpus). Pure functions — no Ollama / corpus — so they're unit-tested
# in CI. Metrics are computed over ANSWERABLE questions only (expected_tools non-empty);
# negatives are scored elsewhere (the deferred generation eval / IDK-rate).


def hit_at_k(retrieved_tools: list[str], expected_tools, k: int) -> int:
    """1 if any expected tool appears in the first k retrieved chunks' tools, else 0."""
    expected = set(expected_tools)
    return int(any(t in expected for t in retrieved_tools[:k]))


def reciprocal_rank(retrieved_tools: list[str], expected_tools) -> float:
    """1/rank of the first retrieved chunk whose tool is acceptable (0 if none in list)."""
    expected = set(expected_tools)
    for i, tool in enumerate(retrieved_tools):
        if tool in expected:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(records, top_k: int = 3) -> dict:
    """records: [{"expected_tools": [...], "retrieved_tools": [...], "type": ...}, ...]
    Returns hit@1, hit@{top_k}, mrr averaged over answerable records."""
    answerable = [r for r in records if r["expected_tools"]]
    n = len(answerable)
    if n == 0:
        return {"n": 0}
    hit1 = sum(hit_at_k(r["retrieved_tools"], r["expected_tools"], 1) for r in answerable)
    hitk = sum(hit_at_k(r["retrieved_tools"], r["expected_tools"], top_k) for r in answerable)
    mrr = sum(reciprocal_rank(r["retrieved_tools"], r["expected_tools"]) for r in answerable)
    return {"n": n, "hit@1": hit1 / n, f"hit@{top_k}": hitk / n, "mrr": mrr / n}
