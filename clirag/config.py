# clirag/config.py
# Single source of truth for pipeline settings. Tune here; nothing else changes.
# Paths are relative to the repo root — run commands from there.

# --- Paths ---
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
CHROMA_DIR = "./chroma_db"
CORPUS_MANIFEST = "data/corpus_manifest.json"

# --- Chunking ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Embedding ---
# CRITICAL: the same model must be used at ingest AND query time. Change this and
# the stored vectors are in a different space than your queries -> delete chroma_db/
# and re-ingest. (corpus_version + this value together define a reproducible index.)
EMBEDDING_MODEL = "nomic-embed-text"   # 768-dim, via Ollama
EMBEDDING_DIM = 768

# --- LLM ---
LLM_MODEL = "gemma3:4b"                # via Ollama
LLM_TEMPERATURE = 0.0                  # deterministic, grounded answers

# --- Retrieval ---
TOP_K = 3

# --- Vector store ---
COLLECTION_NAME = "manpages"
DISTANCE = "cosine"                    # explicit; Chroma defaults to L2 (see DECISIONS.md)
