# main.py
# Responsibility: user-facing CLI entry point.
#
# Two modes:
#   Interactive loop:    python main.py
#   Single question:     python main.py "how do I clone a git repo?"

import sys
from dotenv import load_dotenv
from retrieval import get_chain
from generator import ask

load_dotenv()


def interactive_mode(chain):
    """Keep asking questions until the user types exit/quit/q."""
    print("\nReady. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break
        ask(chain, question)


def single_question_mode(chain, question):
    """Answer one question and exit."""
    ask(chain, question)


def main():
    print("=== Mac RAG Assistant ===\n")
    chain = get_chain()

    if len(sys.argv) > 1:
        # Question passed as command line argument
        # Usage: python main.py "your question here"
        question = " ".join(sys.argv[1:])
        single_question_mode(chain, question)
    else:
        # No argument — drop into interactive loop
        # Usage: python main.py
        interactive_mode(chain)


if __name__ == "__main__":
    main()
