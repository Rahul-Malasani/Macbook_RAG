# clirag/loader.py
# Stage 1 of ingest: read raw man-page .txt files into Document objects.
# Raw implementation — disk -> Document, nothing else. No chunking, no embedding.
import glob
import os

from clirag.config import RAW_DATA_DIR
from clirag.models import Document


def load_documents(raw_dir: str = RAW_DATA_DIR) -> list[Document]:
    """Load every .txt file under `raw_dir` as a Document.

    metadata carries:
      source -> file path (provenance; surfaced in the trace panel later)
      tool   -> command name parsed from "man_<tool>.txt" (used for eval + filtering)
    """
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    docs: list[Document] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        tool = os.path.basename(path).removeprefix("man_").removesuffix(".txt")
        docs.append(Document(text=text, metadata={"source": path, "tool": tool}))
    print(f"[loader] loaded {len(docs)} documents from '{raw_dir}'")
    return docs
