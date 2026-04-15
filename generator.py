# generator.py
# Responsibility: build the prompt, connect retriever + LLM, stream the answer.
# This is the final stage of the retrieval pipeline.

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import LLM_MODEL, LLM_TEMPERATURE


def build_chain(retriever):
    """
    Builds and returns an LCEL chain:
      retriever → prompt → LLM → string output

    The chain takes a question string as input and streams a grounded answer.
    'Answer based only on context' is the key instruction — this is what keeps
    the LLM grounded and prevents hallucination outside your documents.
    """
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions about macOS and Apple documentation.
Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't have that information in my documents."

Context:
{context}

Question: {question}

Answer:""")

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"[generator] Chain ready (model={LLM_MODEL}, temperature={LLM_TEMPERATURE})")
    return chain


def ask(chain, question: str):
    """
    Streams the answer to a question using the built chain.
    Prints chunks as they arrive so you see output immediately.
    """
    print(f"\nQuestion: {question}")
    print("Answer:")
    for chunk in chain.stream(question):
        print(chunk, end="", flush=True)
    print("\n")
