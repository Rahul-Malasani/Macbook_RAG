# Apple Docs RAG — Project Documentation (Source of Truth)

---

## What This Is

A local RAG (Retrieval Augmented Generation) system that lets you ask questions
about your Mac and get accurate, grounded answers from official Apple documentation
and man pages. Fully local — no API costs, no privacy concerns.

---

## Architecture

### Ingest Flow (runs once per new document)
```
raw .txt files
  → loader.py       (load files into Document objects)
  → chunker.py      (split into chunks + save to chunks.json)
  → embedder.py     (configure embedding model)
  → vectorstore.py  (embed chunks + store in ChromaDB)
```

### Retrieval Flow (runs every query)
```
user question
  → embedder.py     (embed the question)
  → vectorstore.py  (load ChromaDB)
  → retriever.py    (find TOP_K most similar chunks)
  → generator.py    (build prompt + stream answer via LLM)
```

---

## Stack

| Component  | Tool                        | Reason                                          |
|------------|-----------------------------|-------------------------------------------------|
| Framework  | LangChain 1.2.15            | Orchestration via LCEL                          |
| Vector DB  | ChromaDB                    | Local, persistent, lightweight                  |
| Embeddings | nomic-embed-text via Ollama | Local, 768 dimensions                           |
| LLM        | Gemma3:4b via Ollama        | Local, working cleanly with LangChain           |
| Tracing    | LangSmith                   | Pipeline observability                          |
| Environment| Python 3.14 + venv          | Isolated dependencies                           |

---

## Project Structure

```
apple-docs-rag/
├── config.py         ← all settings: chunk size, model names, paths, TOP_K
├── loader.py         ← loads raw .txt files into LangChain Document objects
├── chunker.py        ← splits documents into chunks, saves to chunks.json
├── embedder.py       ← configures and returns the embedding model
├── vectorstore.py    ← ChromaDB save (ingest) and load (retrieval)
├── retriever.py      ← similarity search, returns retriever object
├── generator.py      ← LCEL chain, prompt, LLM, streaming
├── ingest.py         ← orchestrator: load → chunk → embed → store
├── retrieval.py      ← orchestrator: load → retrieve → generate
├── main.py           ← CLI entry point (interactive + single question modes)
├── data/
│   ├── raw/          ← source .txt files go here
│   └── processed/
│       └── chunks.json  ← saved chunk checkpoint (auto-generated)
├── chroma_db/        ← vector store (auto-generated, never edit manually)
├── decisions.md      ← full project decision and journey log
├── .env              ← LangSmith API keys (never commit to git)
└── requirements.txt  ← all dependencies
```

---

## Files Explained

### config.py
Single source of truth for all pipeline settings. To tune the pipeline — chunk size,
overlap, model, TOP_K — change values here only. Nothing else needs to be touched.

### loader.py
Walks data/raw/, finds every .txt file, loads each as a LangChain Document object
with .page_content (the text) and .metadata (source filename). No logic beyond loading.

### chunker.py
Splits Document objects using RecursiveCharacterTextSplitter. Saves all chunks to
data/processed/chunks.json as a human-readable checkpoint before embedding.
This lets you inspect exactly what the vector store receives and re-embed with
different settings without reloading raw files.

### embedder.py
Configures and returns the OllamaEmbeddings instance. Kept separate so swapping
embedding models is a one-line change in config.py. The actual embedding computation
happens inside vectorstore.py — this file just sets up the model object.

### vectorstore.py
Two functions: save_to_vectorstore() for ingest (embeds and stores chunks),
load_vectorstore() for retrieval (loads existing ChromaDB). The same embedder
must be passed to both — vectors must live in the same space to be comparable.

### retriever.py
Wraps the vectorstore as a LangChain retriever. Currently uses cosine similarity
search with TOP_K=3. Future experiments: MMR search, hybrid BM25+vector, reranking.

### generator.py
Builds the LCEL chain: retriever → prompt → LLM → StrOutputParser.
The prompt instructs the LLM to answer only from context, which is what keeps
answers grounded and prevents hallucination outside your documents.
Temperature is set to 0.0 for deterministic, factual answers.

### ingest.py
Orchestrator only. Contains no logic — imports and calls loader, chunker, embedder,
vectorstore in sequence. Run this every time new documents are added to data/raw/.

### retrieval.py
Orchestrator only. Wires embedder → vectorstore → retriever → generator.
Builds the chain once; the chain is reused for every question.

### main.py
User-facing CLI. Two modes:
- `python main.py` — interactive loop, ask questions until you type exit
- `python main.py "your question"` — single question mode, answer and exit

---

## Known Issues

| Issue | Status | Fix |
|-------|--------|-----|
| Python 3.14 + Pydantic V1 warning | Active, non-breaking | Recreate venv with Python 3.11 via pyenv if breakage occurs |
| Qwen3.5:4b returns empty content | Active | Qwen uses thinking mode, response lands in different field. Using Gemma3:4b for now |

---

## Current State

- [x] Project structure created
- [x] Virtual environment set up
- [x] All packages installed
- [x] LangSmith connected
- [x] Modular file structure implemented (10 files, single responsibility each)
- [x] Intermediate chunk checkpoint (chunks.json) implemented
- [x] man_git.txt ingested (161 chunks)
- [x] man_curl, man_grep, man_find, man_ssh, man_cp, man_ls ingested (1106 chunks total)
- [x] Retrieval tested — three failure modes observed and documented
- [x] main.py built with interactive + single question CLI modes
- [ ] macOS User Guide epub ingested
- [ ] Document-structure-aware chunking implemented
- [ ] Chunk size tuning experimented with
- [ ] TOP_K tuning experimented with
- [ ] Source display added to answers
- [ ] Hybrid search (BM25 + vector) implemented
- [ ] Qwen thinking mode handled
- [ ] RAGAS evaluation implemented
- [ ] HyDE implemented

---

## Next Steps

### Phase 1 — Better Data (immediate)
1. Download macOS User Guide epub from Apple Books
2. Extract text from epub preserving structure
3. Ingest and test — expect dramatic quality improvement from better data alone

### Phase 2 — Chunking Quality
1. Implement document-structure-aware + recursive chunking for man pages and epub
2. Add metadata to chunks (source filename, section name)
3. Experiment with chunk_size: try 300 and 1000, compare retrieval quality
4. Increase TOP_K to 5, add source display to answers

### Phase 3 — Retrieval Quality
1. Switch to MMR search to reduce redundant chunks
2. Implement hybrid search (BM25 + vector) to cover both semantic and keyword failures
3. Add a reranker (cross-encoder) for TOP_K=20 → rerank → top 3

### Phase 4 — Advanced
1. Fix Qwen thinking mode handling and compare with Gemma3
2. Implement HyDE for better retrieval on natural language questions against man pages
3. Add Apple HIG as structured document source
4. Implement RAGAS evaluation with a test question set
5. Parent-child indexing for epub chapters