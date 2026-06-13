# clirag/chunker.py
# Stage 2 of ingest: split Documents into overlapping Chunks (baseline strategy) and
# checkpoint them to JSON. Raw recursive character splitter — no LangChain.
#
# Baseline = recursive split on a separator hierarchy (paragraph -> line -> word ->
# char) with character overlap. Structure-aware chunking is experiment #3; this is the
# baseline it has to beat. The same splitter is reused at the LangChain migration so
# the parity check stays clean.
import hashlib
import json
import os

from clirag.config import CHUNK_OVERLAP, CHUNK_SIZE, PROCESSED_DATA_DIR
from clirag.models import Chunk, Document

DEFAULT_SEPARATORS = ("\n\n", "\n", " ", "")


def chunk_documents(docs: list[Document], *, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP,
                    corpus_version=None) -> list[Chunk]:
    """Split each Document into Chunks with stable, content-addressed ids.

    The id is deterministic in (source, position, text), so re-running ingest on an
    unchanged corpus produces identical ids — the basis for idempotent upserts.
    """
    _validate(chunk_size, overlap)
    chunks: list[Chunk] = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        tool = doc.metadata.get("tool", "")
        for index, piece in enumerate(split_text(doc.text, chunk_size, overlap)):
            metadata = {
                "source": source,
                "tool": tool,
                "chunk_index": index,
                "char_len": len(piece),
            }
            if corpus_version:
                metadata["corpus_version"] = corpus_version
            chunks.append(
                Chunk(id=_stable_id(source, index, piece), text=piece, metadata=metadata)
            )
    print(f"[chunker] {len(docs)} docs -> {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={overlap})")
    return chunks


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP,
               separators=DEFAULT_SEPARATORS) -> list[str]:
    """Recursively split text on the first applicable separator, then merge the pieces
    back up to chunk_size with `overlap` characters of carry-over."""
    _validate(chunk_size, overlap)
    if not text.strip():
        return []
    return _split(text, chunk_size, overlap, list(separators))


def save_chunks(chunks, path=os.path.join(PROCESSED_DATA_DIR, "chunks.json")) -> None:
    """Write chunks to JSON — a human-readable audit trail of exactly what gets embedded."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = [{"id": c.id, "text": c.text, "metadata": c.metadata} for c in chunks]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[chunker] saved {len(chunks)} chunks to '{path}'")


# --- internals ---

def _validate(chunk_size, overlap):
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")


def _stable_id(source, index, text) -> str:
    key = f"{source}\x1f{index}\x1f{text}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:16]


def _split(text, chunk_size, overlap, separators) -> list[str]:
    # choose the first separator that occurs in `text` ("" always matches, last)
    separator = separators[-1]
    remaining: list[str] = []
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            remaining = separators[i + 1:]
            break

    pieces = list(text) if separator == "" else [s for s in text.split(separator) if s]

    chunks: list[str] = []
    mergeable: list[str] = []
    for piece in pieces:
        if len(piece) < chunk_size:
            mergeable.append(piece)
            continue
        # flush what we've accumulated, then break the oversized piece down further
        if mergeable:
            chunks.extend(_merge(mergeable, separator, chunk_size, overlap))
            mergeable = []
        if remaining:
            chunks.extend(_split(piece, chunk_size, overlap, remaining))
        else:
            chunks.append(piece)  # unsplittable and oversized — keep as-is
    if mergeable:
        chunks.extend(_merge(mergeable, separator, chunk_size, overlap))
    return chunks


def _merge(pieces, separator, chunk_size, overlap) -> list[str]:
    """Greedily combine small pieces up to chunk_size, retaining a trailing window of
    ~overlap characters as the start of the next chunk."""
    sep_len = len(separator)
    chunks: list[str] = []
    window: list[str] = []
    total = 0
    for piece in pieces:
        extra = sep_len if window else 0
        if total + len(piece) + extra > chunk_size and window:
            joined = separator.join(window).strip()
            if joined:
                chunks.append(joined)
            # drop from the front until the carry-over is within `overlap`
            while total > overlap or (
                total + len(piece) + (sep_len if window else 0) > chunk_size and total > 0
            ):
                total -= len(window[0]) + (sep_len if len(window) > 1 else 0)
                window = window[1:]
        window.append(piece)
        total += len(piece) + (sep_len if len(window) > 1 else 0)
    joined = separator.join(window).strip()
    if joined:
        chunks.append(joined)
    return chunks
