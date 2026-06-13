# clirag/query.py
# Orchestrates the query pipeline: embed -> retrieve -> generate. Reuses the persisted
# collection, so run `python -m clirag.ingest` first.
# Run:  python -m clirag.query "how do I copy a file?"     (or no arg for interactive)
import sys

from clirag.embedder import get_embedder
from clirag.generator import get_generator
from clirag.retriever import Retriever
from clirag.vectorstore import get_collection


def build_pipeline():
    embedder = get_embedder()
    retriever = Retriever(embedder, get_collection())
    generator = get_generator()
    return retriever, generator


def answer(query: str, retriever, generator) -> str:
    return generator.generate(query, retriever.retrieve(query))


def main():
    print("=== clirag — offline man-page assistant ===")
    retriever, generator = build_pipeline()
    if len(sys.argv) > 1:
        print("\n" + answer(" ".join(sys.argv[1:]), retriever, generator))
        return
    print("Ask a question (or 'exit').")
    while True:
        try:
            q = input("\nyou> ").strip()
        except EOFError:
            break
        if q.lower() in ("exit", "quit", "q"):
            break
        if q:
            print("\n" + answer(q, retriever, generator))


if __name__ == "__main__":
    main()
