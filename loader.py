# loader.py
# Responsibility: read raw .txt files from disk, return LangChain Document objects.
# Nothing else. No chunking, no embedding, no logic beyond loading.

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from config import RAW_DATA_DIR


def load_documents():
    """
    Walks RAW_DATA_DIR, loads every .txt file as a LangChain Document.
    Each Document has:
      .page_content  → the raw text
      .metadata      → {"source": "data/raw/filename.txt"}
    """
    loader = DirectoryLoader(
        RAW_DATA_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"[loader] Loaded {len(documents)} documents from '{RAW_DATA_DIR}'")
    return documents
