# clirag/models.py
# Lightweight data carriers for the raw pipeline. Deliberately NOT LangChain's
# Document class — in the from-scratch phase we own our own types.
from dataclasses import dataclass, field


@dataclass
class Document:
    """One source document (e.g. a single man page) before chunking."""

    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A chunk of a Document.

    `id` is a stable, content-addressed identifier so re-running ingest is
    idempotent (same content -> same id -> upsert, not duplicate).
    """

    id: str
    text: str
    metadata: dict = field(default_factory=dict)
