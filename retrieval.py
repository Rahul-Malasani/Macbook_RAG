# retrieval.py
# Responsibility: orchestrate the retrieval pipeline.
# Contains NO logic of its own — just wires up the modules for querying.
# Pipeline: load vectorstore → get retriever → build chain → answer question

from dotenv import load_dotenv
from embedder import get_embedder
from vectorstore import load_vectorstore
from retriever import get_retriever
from generator import build_chain, ask

load_dotenv()


def get_chain():
    """
    Builds and returns the full retrieval chain.
    Called once at startup — chain is reused for every question.
    """
    embedder = get_embedder()
    vectorstore = load_vectorstore(embedder)
    retriever = get_retriever(vectorstore)
    chain = build_chain(retriever)
    return chain


if __name__ == "__main__":
    chain = get_chain()
    ask(chain, "how do I clone a git repository?")
