# Project Decision Log

---

## Stack Decisions

### Framework: LangChain
- Abstraction layer for RAG pipeline
- Handles chunking, retrieval, chain composition via LCEL
- Alternative considered: building from scratch
- Why not: learning curve without added benefit at this stage

### Vector Database: ChromaDB
- Local, lightweight, no infra overhead
- Good for development and personal projects
- Alternative considered: FAISS, Pinecone
- Why not FAISS: no persistence. Why not Pinecone: managed/paid, overkill here

### LLMs: Qwen3.5 4b + Gemma3 4b (via Ollama)
- Fully local, no API cost
- Qwen for reasoning-heavy tasks, Gemma as comparison
- Why local: privacy, cost, and learning the stack properly

### Embedding Model: nomic-embed-text (768 dimensions) (via Ollama)
- Lightweight, local
- Strong performance for open source
- CRITICAL: same model must be used at ingest time AND query time — if you change this, delete chroma_db/ and re-ingest

### Data Sources: Apple Docs, HIG, macOS Guidelines, Man Pages
- Structured documents with explicit hierarchy
- Perfect for document-structure-aware chunking
- Personal utility: builds the assistant you actually need

---

## Architectural Decisions

### Pattern: LCEL over RetrievalQA
- LangChain 1.x deprecated RetrievalQA
- Using LCEL pipe syntax instead
- More explicit, composable, and streaming-native

### Model Decision: Gemma3:4b over Qwen3.5:4b for now
- Qwen3.5:4b returns empty content via LangChain due to thinking mode
- Qwen returns response in a different field, needs extra handling
- Using Gemma3:4b until Qwen thinking mode is handled explicitly

### Intermediate Chunk Checkpoint (chunks.json)
- Original ingest.py went directly from chunks to ChromaDB with no saved intermediate
- Decision: save chunks to data/processed/chunks.json before embedding
- Reason: reproducibility, auditability, ability to inspect exactly what the vector store received
- This is the human-readable audit trail of what got embedded

### Modular File Structure over Monolithic ingest.py
- Original project had two files: ingest.py and retrieval.py, each doing everything
- Decision: refactor into single-responsibility modules
- config.py — all settings (chunk size, model names, paths, TOP_K)
- loader.py — loads raw .txt files into LangChain Document objects
- chunker.py — splits documents into chunks, saves to chunks.json
- embedder.py — configures and returns the embedding model
- vectorstore.py — handles ChromaDB save and load separately
- retriever.py — configures similarity search, returns retriever object
- generator.py — builds LCEL chain, streams answers
- ingest.py — orchestrator only, no logic, calls modules in order
- retrieval.py — orchestrator only, no logic, wires retrieval modules
- main.py — CLI entry point with both interactive and single-question modes
- Reason: each module has one reason to change. Swapping embedding model, tuning chunk size, or changing LLM is now a single-line change in config.py

### CLI Design: Dual Mode
- Decision: main.py supports both interactive loop and single question mode
- Usage: `python main.py` for interactive, `python main.py "question"` for single shot
- Reason: interactive is better for exploration, single question is better for scripting and testing

---

## Known Issues

### Python 3.14 + LangChain
- LangChain uses Pydantic V1 internally which isn't fully compatible with 3.14
- Currently showing warning but not breaking
- If unexplained errors appear, recreate venv with Python 3.11 via pyenv

### Qwen3.5:4b Empty Content
- Qwen uses thinking mode, response lands in a different field than standard content
- LangChain's default parser reads the wrong field, gets empty string
- Fix: handle thinking tokens explicitly before re-enabling Qwen

---

## Retrieval Failure Observations (from live testing)

This section documents real retrieval failures observed during development.
These are not bugs — they are expected failure modes of basic flat chunking
on man pages, and understanding them is the reason we're moving to better
data sources and advanced chunking strategies.

### Test 1: "how do I copy a file?" → cp man page
- Expected: basic usage like `cp file.txt destination/`
- Got: raw SYNOPSIS line with full flag matrix
- Why: the synopsis section is dense with the command name, scores high on similarity,
  but is the least useful section for a natural language question

### Test 2: "how do I find all files modified in the last 7 days?" → find man page
- Expected: example using `-mtime -7`
- Got: "I don't have that information in my documents"
- Why: classic semantic gap — the user's natural language phrasing and the man page's
  technical language are too far apart for basic cosine similarity to bridge.
  The embedding of "modified in the last 7 days" did not match the embedding of
  "-mtime n: True if the difference between the file last modification time" closely enough.

### Test 3: "how do I connect to a remote server?" → expected ssh, got curl
- Expected: ssh basic usage
- Got: curl's --connect-to flag explanation
- Why: the word "connect" matched a curl option more closely than ssh's basic usage section.
  Wrong document retrieved entirely. This is a vocabulary collision problem.

### Root Cause of All Three Failures
Man pages are written for people who already know the command. The writing style
is technical, dense, and defensive. The semantic gap between how users ask questions
and how man pages describe answers is too large for basic similarity search to bridge
without additional techniques like HyDE (Hypothetical Document Embeddings).

### Decision: Move to Apple epub as primary data source
- Man pages kept for technical flag lookups but not relied on for natural language Q&A
- macOS User Guide epub is written in plain language for users — far better semantic match
- This is the single most impactful improvement available without changing any pipeline code

---

## Chunking Strategy Decisions

### Current: Flat RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
- Simple, works for initial testing
- Known weakness: blind to document structure, splits across section and option boundaries
- Result: chunks that start mid-explanation of one flag and end mid-explanation of another

### Planned: Document-Structure-Aware + Recursive (Phase 2)
- Man pages have consistent structure: NAME, SYNOPSIS, DESCRIPTION, OPTIONS, EXAMPLES
- The section is the natural chunk unit, not a fixed character count
- Strategy: split by section headers first (structure-aware), then apply recursive
  splitting within each section for size control (prevents context rot in long sections)
- This combination respects meaning boundaries while still enforcing size limits
- Prerequisite: must be implemented at chunking stage — cannot retrofit existing flat chunks

---

## Next Planned Experiments (in order)

### Experiment 1 — Better Data (immediate)
Get macOS User Guide epub from Apple Books, extract text, re-ingest.
Expected outcome: dramatic improvement in answer quality purely from better data,
no code changes required.

### Experiment 2 — Chunk Size Tuning
Try chunk_size=300 and chunk_size=1000, compare retrieval quality on same questions.
Smaller chunks = more precise retrieval but risk losing context.
Larger chunks = more context but risk bringing in irrelevant content that confuses LLM.

### Experiment 3 — Increase TOP_K + Add Source Display
Retrieve 5 chunks instead of 3, show source filename alongside answer.
Helps diagnose retrieval problems faster and gives user provenance of the answer.

### Experiment 4 — Document-Structure-Aware Chunking
Implement section-aware splitting for man pages and epub chapters.
Measure whether retrieval quality improves on the three failed test questions above.

### Experiment 5 — MMR Search
Switch retriever from similarity to Maximum Marginal Relevance.
MMR penalises redundant chunks — if two chunks say nearly the same thing,
it picks one and finds a more diverse third chunk instead.
Useful when TOP_K chunks are all from the same section of a document.

### Experiment 6 — Hybrid Search (BM25 + Vector)
Add keyword search alongside semantic search.
Keyword search excels at exact technical terms (flag names, command syntax).
Vector search excels at natural language meaning.
Combining both covers both failure modes observed in testing.

### Experiment 7 — HyDE (Hypothetical Document Embeddings)
Instead of embedding the user question directly, ask the LLM to generate a
hypothetical answer first, then embed that. A hypothetical answer written in
man-page style will match man-page chunks much more closely than a natural
language question would.

### Experiment 8 — RAGAS Evaluation
Build a small test set of question-answer pairs with known correct answers.
Run RAGAS metrics: faithfulness, answer relevancy, context precision, context recall.
This turns subjective "does this seem right" into measurable scores across experiments.