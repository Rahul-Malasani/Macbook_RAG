# config.py
# Single source of truth for all pipeline settings.
# To experiment, change values here — nothing else needs to touch.

# --- Paths ---
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
CHROMA_DIR = "./chroma_db"

# --- Chunking ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Embedding ---
EMBEDDING_MODEL = "nomic-embed-text"  # must match at ingest AND query time

# --- LLM ---
LLM_MODEL = "gemma3:4b"
LLM_TEMPERATURE = 0.0  # 0 = deterministic, grounded answers. Increase for creativity.

# --- Retrieval ---
TOP_K = 3  # number of chunks to retrieve per query
