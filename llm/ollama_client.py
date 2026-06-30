from llama_index.llms.ollama import Ollama

from config.settings import (
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT
)

# =====================================================
# Cached Ollama instance
# =====================================================

_ollama = None


def get_ollama():

    global _ollama

    if _ollama is None:

        print(
            f"[OLLAMA] Loading model: {OLLAMA_MODEL}"
        )

        _ollama = Ollama(

            model=OLLAMA_MODEL,

            request_timeout=OLLAMA_TIMEOUT,

            temperature=0.0,

            context_window=8192
        )

        print(
            "[OLLAMA] Ready."
        )

    return _ollama


class OllamaClient:

    def __init__(self):

        # Reuse cached Ollama instance.
        self.llm = get_ollama()

    def generate(
        self,
        prompt: str
    ):

        response = self.llm.complete(
            prompt
        )

        return str(response)
