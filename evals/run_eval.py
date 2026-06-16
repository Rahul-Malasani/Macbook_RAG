# evals/run_eval.py
# Run the retrieval eval over the golden set and append ONE row to the ablation table
# (evals/results/experiments.jsonl) — every stack/experiment adds a row, reproducibly
# tagged with stack + git_sha + corpus_version + config.
# Run:  python -m evals.run_eval [--stack raw|langchain]
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from clirag.config import (
    CHUNK_OVERLAP, CHUNK_SIZE, CORPUS_MANIFEST, DISTANCE, EMBEDDING_MODEL, TOP_K,
)
from evals.retrieval_eval import evaluate, hit_at_k

GOLDEN = "evals/golden/golden_set.jsonl"
RESULTS = "evals/results/experiments.jsonl"


def load_golden(path=GOLDEN):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_retriever(stack, top_k):
    """Return an object exposing .retrieve(query) -> [{'metadata': {'tool': ...}}, ...]."""
    if stack == "langchain":
        from clirag.lc import LCEvalRetriever     # lazy: keeps langchain out of the raw path
        return LCEvalRetriever(top_k)
    from clirag.embedder import get_embedder
    from clirag.retriever import Retriever
    from clirag.vectorstore import get_collection
    return Retriever(get_embedder(), get_collection(), top_k=top_k)


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def _corpus_version():
    try:
        with open(CORPUS_MANIFEST) as f:
            return json.load(f)["corpus_version"]
    except Exception:
        return None


def run(top_k=TOP_K, stack="raw"):
    golden = load_golden()
    retriever = _build_retriever(stack, top_k)

    records = []
    for q in golden:
        hits = retriever.retrieve(q["question"])
        records.append({
            "id": q["id"],
            "type": q["type"],
            "expected_tools": q["expected_tools"],
            "retrieved_tools": [h["metadata"]["tool"] for h in hits],
        })

    metrics = evaluate(records, top_k=top_k)
    by_type = _hit_by_type(records, top_k)
    misses = [r for r in records
              if r["expected_tools"] and not hit_at_k(r["retrieved_tools"], r["expected_tools"], top_k)]

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stack": stack,
        "git_sha": _git_sha(),
        "corpus_version": _corpus_version(),
        "config": {
            "chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP, "top_k": top_k,
            "embedding_model": EMBEDDING_MODEL, "distance": DISTANCE,
        },
        "metrics": metrics,
        "by_type": by_type,
    }
    _append(row)

    print(f"\n=== retrieval [{stack}] (n={metrics['n']} answerable) ===")
    for key in ("hit@1", f"hit@{top_k}", "mrr"):
        print(f"  {key:7} {metrics[key]:.3f}")
    print(f"  hit@{top_k} by type: {by_type}")
    print(f"  misses ({len(misses)}):")
    for m in misses:
        print(f"    [{m['id']}] {m['type']:9} expected {m['expected_tools']} -> got {m['retrieved_tools']}")
    return row


def _hit_by_type(records, top_k):
    buckets = {}
    for r in records:
        if r["expected_tools"]:
            buckets.setdefault(r["type"], []).append(
                hit_at_k(r["retrieved_tools"], r["expected_tools"], top_k))
    return {t: round(sum(v) / len(v), 3) for t, v in buckets.items()}


def _append(row, path=RESULTS):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[eval] appended row to {path} (stack={row['stack']}, git={row['git_sha']})")


if __name__ == "__main__":
    stack = "raw"
    if "--stack" in sys.argv:
        stack = sys.argv[sys.argv.index("--stack") + 1]
    run(stack=stack)
