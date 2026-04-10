# Project Decision Log

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

### Embedding Model: nomic-embed-text(768 dimensions) (via Ollama)
- Lightweight, local
- Strong performance for open source

### Data Sources: Apple Docs, HIG, macOS Guidelines, Man Pages
- Structured documents with explicit hierarchy
- Perfect for document-structure-aware chunking
- Personal utility: builds the assistant you actually need




























## Known Issue: Python 3.14 + LangChain
- LangChain uses Pydantic V1 internally which isn't fully compatible with 3.14
- Currently showing warning but not breaking
- If unexplained errors appear, recreate venv with Python 3.11 via pyenv