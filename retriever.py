# retriever.py
# Responsibility: take a vectorstore, return a retriever object that fetches
# the TOP_K most semantically similar chunks for any given query.
# This is where you'll later add reranking, hybrid search, metadata filtering.

from config import TOP_K


def get_retriever(vectorstore):
    """
    Returns a LangChain retriever from the vectorstore.
    search_type="similarity" = standard cosine similarity search (bi-encoder).
    k = number of chunks to retrieve — tunable in config.py.

    Future experiments to try here:
      - Increase k and add a reranker (cross-encoder) to rerank before sending to LLM
      - search_type="mmr" for Maximum Marginal Relevance (reduces redundant chunks)
      - Add metadata filters e.g. filter by source document
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )
    print(f"[retriever] Retriever ready (top_k={TOP_K})")
    return retriever
