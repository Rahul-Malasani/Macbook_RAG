# clirag — offline CLI / man-page assistant (RAG)

A local Retrieval-Augmented-Generation system that answers natural-language questions
about Unix command-line tools, grounded in their man pages. Fully offline during
development (Ollama + ChromaDB); deployed with a hosted LLM.

> **Status: in development.** This repo is built as an *engineering* artifact, not a
> demo — the point is the method (raw-first, eval-driven, measured experiments), not
> just a working chatbot. See [DECISIONS.md](DECISIONS.md) for the reasoning log.

## Approach

1. **Build the pipeline raw** (no framework) to understand every layer.
2. **Evaluate to a baseline** — golden set + retrieval metrics + generation metrics.
3. **Migrate to LangChain**, confirm parity, record the ergonomics/latency cost.
4. **Improve one experiment at a time**, each measured against a fixed corpus version,
   each with a one-line tradeoff. Stop at diminishing returns.
5. **Harden, package, deploy** with a live trace + metrics + ablation dashboard.

## Stack (lean by design)

| Component   | Choice                          | Notes                                   |
|-------------|---------------------------------|-----------------------------------------|
| Vector DB   | ChromaDB (persistent, local)    | cosine distance (explicit, not default) |
| Embeddings  | `nomic-embed-text` via Ollama   | 768-dim; swapped to an API at deploy    |
| LLM         | `gemma3:4b` via Ollama          | swapped to an API at deploy             |
| Lang        | Python ≥3.11                    | packaged (`pyproject.toml`)             |

## Quickstart

Requires a local [Ollama](https://ollama.com) with `nomic-embed-text` and `gemma3:4b` pulled.

```bash
pip install -e .                                  # or: venv/bin/pip install -e ".[dev]"
python -m clirag.ingest                           # load → chunk → embed → store (6,485 chunks)
python -m clirag.query "how do I copy a file?"    # retrieve → generate (or no arg = interactive)
```

## Corpus

A **curated, reproducible** set of common Unix CLI tools (currently **128 man pages**),
chosen with deliberate *confusable clusters* (`grep/egrep/fgrep`, `scp/rsync/sftp`,
`find/locate/fd`) so retrieval has to disambiguate — which gives the ablation study
headroom. The corpus is content-addressed by `corpus_version` in
[`data/corpus_manifest.json`](data/corpus_manifest.json); every eval result is tagged
with it.

```bash
python3 scripts/harvest_manpages.py   # regenerate corpus + manifest from the candidate list
```

## Layout

```
clirag/        config · models · loader · chunker · embedder · vectorstore · retriever · generator · ingest · query
evals/         golden set + retrieval/generation eval + experiments table  (step 2)
scripts/       corpus harvesting
tests/         unit tests (run in CI)
data/raw/      man_<tool>.txt  +  corpus_manifest.json
app/           FastAPI + dashboard  (step 7)
```

## Roadmap

- [x] **0 — Foundation:** package layout, config, corpus harvest + manifest, CI + tests, docs
- [x] **1 — Raw pipeline:** load → chunk → embed → store → retrieve → generate (end-to-end on 128 pages)
- [x] **2 — Eval harness:** 50-question golden set + retrieval metrics (hit@k, MRR) → baseline (hit@3 0.69); generation metrics deferred
- [ ] **3 — Telemetry:** per-query trace store (chunks+scores, per-stage latency, tokens)
- [ ] **4 — LangChain migration** + parity check + cost-of-framework writeup
- [ ] **5 — Experiments:** structure-aware chunking · reranker · hybrid · query-rewrite · MMR · threshold · cache
- [ ] **6 — Hardening:** timeouts/retries, relevance threshold, graceful degradation, rate limit
- [ ] **7 — Deploy:** API LLM/embeddings, Docker, FastAPI, live dashboard
- [ ] **8 — Multi-format ingestion:** macOS User Guide epub / PDFs / tables (documented extension)

## Limitations (current)

Single-turn only. English only. Man-page corpus is dense/technical (a known retrieval
challenge — see the failure observations in [DECISIONS.md](DECISIONS.md)). Not yet
load-tested. Eval set and LLM-judge metrics are noisy by nature.
