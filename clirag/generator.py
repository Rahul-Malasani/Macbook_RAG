# clirag/generator.py
# Final pipeline stage: build a grounded prompt from retrieved chunks and answer with
# Ollama (gemma3:4b). Raw client — no LangChain.
#
# Baseline is deliberately minimal: returns the answer string only. Token counts,
# latency, and streaming come later (telemetry / deploy) — after the experiments.
import ollama

from clirag.config import LLM_MODEL, LLM_TEMPERATURE

IDK = "I don't have that information in my documents."

SYSTEM = (
    "You answer questions about Unix command-line tools using ONLY the provided context "
    "from man pages. If the answer is not in the context, reply exactly: "
    f'"{IDK}" Do not use outside knowledge.'
)


class Generator:
    def __init__(self, model: str = LLM_MODEL, temperature: float = LLM_TEMPERATURE):
        self.model = model
        self.temperature = temperature

    def generate(self, query: str, hits: list[dict]) -> str:
        if not hits:                       # nothing retrieved -> don't hallucinate
            return IDK
        try:
            resp = ollama.chat(
                model=self.model,
                messages=self._build_messages(query, hits),
                options={"temperature": self.temperature},
            )
        except Exception as e:             # broad on purpose; retries are step 6
            raise RuntimeError(
                f"Ollama chat failed for '{self.model}'. Is `ollama serve` running "
                f"and `{self.model}` pulled? ({e})"
            ) from e
        message = getattr(resp, "message", None)
        text = message.content if message is not None else resp["message"]["content"]
        return text.strip()

    def _build_messages(self, query: str, hits: list[dict]) -> list[dict]:
        context = "\n\n".join(f"[{h['metadata'].get('tool', '?')}] {h['text']}" for h in hits)
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]


def get_generator() -> Generator:
    print(f"[generator] model={LLM_MODEL} (temperature={LLM_TEMPERATURE})")
    return Generator()
