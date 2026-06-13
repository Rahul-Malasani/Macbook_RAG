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

## Chunking strategy  *(migrated)*

- **Current:** flat `RecursiveCharacterTextSplitter` (size 500, overlap 50). Simple; blind
  to structure → splits across section/option boundaries.
- **Planned:** structure-aware (split on man-page sections: NAME/SYNOPSIS/DESCRIPTION/
  OPTIONS/EXAMPLES) then recursive within a section for size control. Must be done at
  chunk time — cannot retrofit existing flat chunks.

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
