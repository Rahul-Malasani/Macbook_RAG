# clirag/lc.py
# LangChain implementation of the same pipeline (migration phase, step 4). Same model,
# same chunk size/overlap, same cosine metric, same top_k, same prompt as the raw
# pipeline — so retrieval metrics should land on the raw baseline. The deliverable is the
# framework comparison (LOC / deps / ergonomics), not new numbers.
#
# Builds into a SEPARATE collection (chroma_db_lc) so the raw index stays intact for a
# side-by-side eval. Requires the optional extra:  pip install -e ".[langchain]"
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from clirag.config import (
    CHUNK_OVERLAP, CHUNK_SIZE, DISTANCE, EMBEDDING_MODEL, LLM_MODEL, LLM_TEMPERATURE,
    RAW_DATA_DIR, TOP_K,
)
from clirag.generator import SYSTEM   # reuse the EXACT same grounding prompt

LC_CHROMA_DIR = "./chroma_db_lc"
LC_COLLECTION = "manpages_lc"


def _embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def lc_ingest():
    """Load -> split (RecursiveCharacterTextSplitter) -> embed -> store, the LangChain way."""
    docs = DirectoryLoader(RAW_DATA_DIR, glob="*.txt", loader_cls=TextLoader).load()
    for d in docs:   # parity with the raw loader's `tool` metadata
        name = os.path.basename(d.metadata.get("source", ""))
        d.metadata["tool"] = name.removeprefix("man_").removesuffix(".txt")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    Chroma.from_documents(
        chunks, embedding=_embeddings(),
        persist_directory=LC_CHROMA_DIR, collection_name=LC_COLLECTION,
        collection_metadata={"hnsw:space": DISTANCE},
    )
    print(f"[lc] ingested {len(chunks)} chunks into '{LC_COLLECTION}'")
    return len(chunks)


def lc_retriever(top_k=TOP_K):
    store = Chroma(persist_directory=LC_CHROMA_DIR, collection_name=LC_COLLECTION,
                   embedding_function=_embeddings())
    return store.as_retriever(search_kwargs={"k": top_k})


def lc_chain(top_k=TOP_K):
    """LCEL: retriever -> prompt -> ChatOllama -> str. Same prompt as the raw generator."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    def format_docs(docs):
        return "\n\n".join(f"[{d.metadata.get('tool', '?')}] {d.page_content}" for d in docs)

    return (
        {"context": lc_retriever(top_k) | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )


class LCEvalRetriever:
    """Adapter so evals/run_eval can read retrieved tools from the LangChain retriever
    through the same .retrieve(query) -> [{'metadata': {'tool': ...}}] interface."""

    def __init__(self, top_k=TOP_K):
        self._retriever = lc_retriever(top_k)

    def retrieve(self, query: str) -> list[dict]:
        docs = self._retriever.invoke(query)
        return [{"metadata": {"tool": d.metadata.get("tool")}, "text": d.page_content} for d in docs]


if __name__ == "__main__":
    lc_ingest()
