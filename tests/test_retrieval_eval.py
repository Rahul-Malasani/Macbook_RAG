from evals.retrieval_eval import evaluate, hit_at_k, reciprocal_rank


def test_hit_at_k():
    assert hit_at_k(["cp", "dd", "ls"], ["cp"], 1) == 1
    assert hit_at_k(["dd", "cp", "ls"], ["cp"], 1) == 0          # cp at rank 2, not top-1
    assert hit_at_k(["dd", "cp", "ls"], ["cp"], 3) == 1
    assert hit_at_k(["ls", "dd"], ["cp"], 3) == 0
    assert hit_at_k(["scp", "x"], ["scp", "rsync", "sftp"], 1) == 1   # acceptable set


def test_reciprocal_rank():
    assert reciprocal_rank(["cp", "dd"], ["cp"]) == 1.0
    assert reciprocal_rank(["dd", "cp"], ["cp"]) == 0.5
    assert reciprocal_rank(["dd", "ls"], ["cp"]) == 0.0


def test_evaluate_excludes_negatives():
    records = [
        {"expected_tools": ["cp"], "retrieved_tools": ["cp"], "type": "lookup"},
        {"expected_tools": ["find"], "retrieved_tools": ["ls"], "type": "lookup"},
        {"expected_tools": [], "retrieved_tools": ["curl"], "type": "negative"},   # ignored
    ]
    m = evaluate(records, top_k=3)
    assert m["n"] == 2                       # negative excluded
    assert m["hit@1"] == 0.5
    assert m["hit@3"] == 0.5
    assert m["mrr"] == 0.5                   # (1.0 + 0.0) / 2
