# Decision Log

Why things are the way they are — including alternatives considered and rejected, and
decisions later reversed (dated). Toys contain only what's there; this records what was
weighed and declined.

---

## Method decisions (2026-06-13)

### Build raw first, then migrate to LangChain  *(reverses an earlier decision)*
- **Earlier decision:** use LangChain from the start; "building from scratch — learning
  curve without added benefit at this stage."
- **Reversed to:** build the pipeline from scratch (no framework) first, establish a
  baseline with evals, *then* migrate to LangChain as an explicit phase.
- **Why:** the point of this project is to understand every layer well enough to defend
  it. A raw build does that; it also makes the custom trace/observability layer coherent
  (instrumenting our own code) and gives a real before/after for the framework migration.
- **What the migration must prove:** *not* better eval numbers (same model/chunks/prompt
  → numbers land on the baseline; if they move, it's a bug). It proves the framework's
  cost/benefit: lines of code, latency overhead, observability friction.

### Corpus: curated & reproducible, not "all man pages"
- **Decision:** a defined candidate list (~120–150 common CLI tools), harvested into
  `data/raw/` with a content-hash `corpus_version` in `data/corpus_manifest.json`.
  Currently 128 pages.
- **Alternative considered:** `man -k .` / every installed page. **Rejected:** machine-
  dependent (non-reproducible), uncontrolled variance (huge `bash`/`zsh` pages dominate),
  not describable on a resume.
- **Deliberate confusable clusters** (`grep/egrep/fgrep`, `scp/rsync/sftp`,
  `find/locate/fd`, archive family) so retrieval has to disambiguate → the ablation has
  headroom. A corpus of fully-distinct docs would sit at the retrieval ceiling and no
  strategy would show movement.

### Homogeneous corpus first; multi-format later
- **Decision:** v1 is man pages only (uniform plain text). Multi-format ingestion
  (macOS User Guide epub / PDFs / tables) is a **separate, later, documented phase**.
- **Why:** uniform text isolates the retrieval/generation ablation from data-parsing
  confounds — a retrieval miss is then a *strategy* failure, not a bad PDF extraction.
  Multi-format isn't dropped; it's sequenced, because parsing is its own marketable
  competency and deserves its own before/after.

### Goal is the knee of the curve, not "near-perfect"
- **Decision:** stop adding strategies when marginal gain drops below the eval set's
  noise floor, and document why. Ship a defensible v1, then iterate.
- **Why:** chasing the last points on a ~50-question LLM-judged set is chasing noise and
  overfitting the eval set — the opposite of the intended signal.

### Packaging & structure
- `clirag/` importable package (clean imports for tests/Docker/FastAPI) over flat modules.
- `pyproject.toml` over `requirements.txt` (deps + ruff/pytest config in one place).
- Foundation stood up *early and reused*: versioned ingest, eval harness as a regression
  suite, telemetry, CI + tests, this log — not bolted on at the end.

### Vector store: cosine distance (explicit)
- **Decision:** create the Chroma collection with `hnsw:space = cosine`.
- **Why:** embeddings are compared by direction, not magnitude; cosine is the right metric.
  Chroma defaults to L2 — leaving the default would silently use a metric we didn't intend.

### Embeddings: no task prefixes in the baseline
- **Decision:** the baseline embeds raw text with no instruction prefix.
- `nomic-embed-text` is trained with `search_document:` / `search_query:` prefixes that
  usually lift retrieval. `embed_documents` / `embed_query` are split so adding them is a
  one-line change — **"add nomic task prefixes" is a measured experiment, not baked in.**

### Generation baseline is minimal (tokens/latency/streaming deferred)
- **Decision:** the baseline generator returns only the answer string. Token counts,
  latency, and streaming are added later (telemetry / deploy), *after* the experiments —
  they don't change retrieval/generation quality, only observability.

---

## Stack decisions  *(migrated)*

### Vector DB: ChromaDB
- Local, lightweight, persistent, no infra overhead.
- **Rejected:** FAISS (no persistence), Pinecone (managed/paid, overkill at this scale).
  Would revisit a hosted store (e.g. Qdrant/pgvector) only past single-node scale.

### Embeddings: `nomic-embed-text` (768-dim) via Ollama
- Local, strong open-source quality. Same model required at ingest **and** query time.

### LLM: `gemma3:4b` via Ollama (local); API at deploy
- Fully local during dev (no cost, privacy). Swapped to a hosted API at deploy because
  Ollama isn't cheap to host — cost metrics and latency get re-measured after the swap.

### Chunk checkpoint (`data/processed/chunks.json`)
- Ingest writes chunks to JSON before embedding — a human-readable audit trail of exactly
  what got embedded; supports re-embedding without reloading raw files.

### (LangChain phase) LCEL over deprecated RetrievalQA
- For the migration phase: explicit, composable, streaming-native pipe syntax.

---

## Retrieval failure observations (from live testing)  *(migrated — evidence)*

Real failures of basic flat chunking + cosine similarity on man pages. Not bugs — the
expected failure modes that motivate the experiments.

### Test 1: "how do I copy a file?" → cp man page
- Expected basic usage (`cp file.txt destination/`); got the raw SYNOPSIS flag matrix.
- The synopsis is dense with the command name → scores high on similarity, but is the
  least useful section for a natural-language question. → motivates **structure-aware chunking**.

### Test 2: "how do I find all files modified in the last 7 days?" → find man page
- Expected `-mtime -7`; got "I don't have that information."
- Classic **semantic gap**: natural phrasing vs. terse technical language
  ("-mtime n: difference between the file last modification time…"). → motivates **HyDE**.

### Test 3: "how do I connect to a remote server?" → expected ssh, got curl
- "connect" matched a curl option more closely than ssh's basic usage. Wrong document
  entirely — a **vocabulary collision**. → motivates **hybrid (BM25 + vector)**.

### Root cause
Man pages are written for people who already know the command — dense, technical, defensive.
The gap between how users ask and how man pages answer is too wide for plain similarity.

---

## Chunking strategy

- **Current (baseline):** our own raw recursive character splitter (size 500, overlap 50;
  separator hierarchy paragraph→line→word→char) with stable content-hash chunk ids for
  idempotent ingest. No LangChain. Simple; blind to structure → splits across
  section/option boundaries. (Reused at the LangChain migration so the parity check is clean.)
- **Planned:** structure-aware (split on man-page sections: NAME/SYNOPSIS/DESCRIPTION/
  OPTIONS/EXAMPLES) then recursive within a section for size control. Must be done at
  chunk time — cannot retrofit existing flat chunks.

---

## Corpus observations (2026-06-13)

### Near-duplicate documents
`man apropos`, `man whatis`, and `man man` resolve to the same MAN(1) page, so the corpus
holds a few near-identical documents. A realistic retrieval wrinkle (near-dupes compete
for one query) — relevant to MMR (experiment 6) and to how retrieval hits are counted.

### Header/footer boilerplate in chunks
Every page repeats a running header/footer (`MAN(1)  General Commands Manual  MAN(1)`),
which leaks into chunks as low-value tokens. Candidate cleanup at chunk time — see backlog.

---

## Baseline end-to-end run (2026-06-13, corpus_version 79dea7c7938d)

First full raw pipeline (load→chunk→embed→store→retrieve→generate), top_k=3, gemma3:4b.
Grounding works — no hallucination: the negative control and the retrieval misses both
returned the IDK answer rather than inventing one. The known failure modes reproduced;
this is the headroom the experiments target:

- **"copy a file?"** → right page (cp), but the SYNOPSIS flag line, not basic usage. (Test 1)
- **"files modified in the last 7 days?"** → `find` not even in top-k (got ls/readlink/stat)
  → IDK. Retrieval miss from the semantic gap. (Test 2)
- **"connect to a remote server?"** → curl ranked #1 over ssh (vocabulary collision, Test 3);
  ssh was in top-k, so the answer came from ssh (port-forwarding, not basic usage).
- **"make a file executable?"** → chmod not retrieved at all; rsync's "executability" option
  dominated → wrong answer. New data point for hybrid / HyDE.
- **"capital of France?"** → IDK (negative control passes).

Most errors are **retrieval** failures (wrong/missing top-k), not generation — so the eval
must score retrieval and generation separately (step 2).

---

## Baseline retrieval metrics (2026-06-13, corpus_version 79dea7c7938d, commit c560212)

Golden set: 50 hand-authored questions (42 answerable + 8 negatives), tool-level labels.
top_k=3, nomic-embed-text, cosine. First row in `evals/results/experiments.jsonl`.

| metric | value |
|--------|-------|
| hit@1  | 0.476 |
| hit@3  | 0.690 |
| MRR    | 0.567 |

hit@3 by type: lookup 0.70 · ambiguous 0.67 · multi-hop 0.67. **13/42 answerable misses** —
natural-language queries vs terse man-page wording (`"rename a file"`→mv missed;
`"count lines"`→wc missed; `"search for a word"`→grep missed; chmod/find missed). This is the
headroom the experiments target (hybrid/BM25, HyDE, structure-aware chunking, task prefixes).
Some labels are strict (e.g. q036 excludes `compress`) — pending golden-set review.

Generation metrics deferred (per scope) — retrieval is where every observed failure lives.

---

## LangChain migration (step 4 — 2026-06-13)

Rebuilt the pipeline on LangChain (`clirag/lc.py`): `DirectoryLoader` →
`RecursiveCharacterTextSplitter` (500/50) → `OllamaEmbeddings` → `langchain-chroma`
(cosine) → retriever (k=3) → LCEL chain (`ChatOllama` gemma3:4b, same prompt). Built into a
separate collection (`manpages_lc`); the raw index is untouched for side-by-side eval.

**Behavior — preserved** (`--stack langchain` row in `experiments.jsonl`):

| stack | hit@1 | hit@3 | MRR | misses |
|-------|-------|-------|-----|--------|
| raw       | 0.476 | 0.690 | 0.567 | 13 |
| langchain | 0.452 | 0.690 | 0.556 | 13 (identical set) |

hit@3 identical and the **13 missed questions are the same set** → behavior-preserving. The
hit@1/MRR dip is one question changing rank, from `RecursiveCharacterTextSplitter` producing
**6,507 chunks vs my reimplementation's 6,485** (`keep_separator` differs). A chunker-impl
artifact, not a retrieval regression — which is exactly the kind of move the log demanded we explain.

**Cost / benefit (the real deliverable):**
- **Wiring LOC: 452 → 89 (−80%).** RCTS replaces the ~130-line recursive splitter + merge;
  `Chroma.from_documents` replaces the idempotent batched upsert; `as_retriever` + LCEL
  replace the retriever/generator glue.
- **Dependencies: 2 → 8 direct** (+`langchain`, `-core`, `-community`, `-chroma`, `-ollama`,
  `-text-splitters`), heavy transitive tree (~116 pkgs in the venv); carries the Pydantic-v1
  vs Python-3.14 warning.
- **Control lost:** retriever yields LangChain `Document`s (needed an adapter for the eval);
  stable chunk IDs, delete-by-source idempotency, and per-stage instrumentation are no longer
  ours to shape — the trace panel will hook LangChain **callbacks** instead.

**Verdict:** keep LangChain as the base for the experiments (faster to swap retrievers /
rerankers), with the raw pipeline frozen as the "I understand the internals" reference.

---

## Experiment backlog (each measured vs. a fixed corpus_version, each with a tradeoff line)

1. Chunk-size / overlap sweep (300 / 500 / 1000)
2. TOP_K sweep + source/provenance display
3. **Structure-aware chunking** (targets Test 1)
4. **Hybrid search** BM25 + vector (targets Test 3)
5. **HyDE** (targets Test 2)
6. MMR (reduce redundant chunks)
7. Reranker (cross-encoder over a wider candidate set)
8. Relevance threshold tuning ("I don't know" path)
9. Semantic caching (latency/cost; guard correctness)
10. Embedding task prefixes (nomic `search_query:` / `search_document:`)
11. Strip man-page header/footer boilerplate at chunk time
> Eval harness (golden set + retrieval/generation metrics, RAGAS-style) is **step 2** —
> built before experiment 1 so every row above has numbers.

---

## Known issues  *(migrated)*

- **Python 3.14 + Pydantic v1** warning under LangChain — non-breaking; recreate venv with
  3.11 via pyenv if it breaks. (Irrelevant to the raw phase — no LangChain there.)
- **Qwen3.5:4b empty content** via LangChain (thinking-mode response lands in a different
  field). Using `gemma3:4b` until thinking tokens are handled explicitly.

---

## Reference
- LangChain reference implementation (pre-restructure) preserved at commit `d53768f`,
  branch `legacy-langchain` — the migration-comparison baseline (step 4).
