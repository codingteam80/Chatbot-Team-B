from llama_index.llms.ollama import Ollama

from config.settings import (
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT
)


class OllamaClient:

    def __init__(self):

        # Initialize Ollama LLM connection
        self.llm = Ollama(

            # Model name from settings.py
            # Example: llama3.2:3b
            model=OLLAMA_MODEL,

            # Maximum waiting time for response
            # Current value: 120 seconds
            request_timeout=OLLAMA_TIMEOUT,

            # Lower temperature = more consistent answers
            # 0.0 = deterministic output
            temperature=0.0,

            # Maximum context size sent to model
            # Current value: 8192 tokens
            context_window=8192
        )

    def generate(
        self,
        prompt: str
    ):

        # Send prompt to Ollama model
        response = self.llm.complete(
            prompt
        )

        # Convert response object to string
        return str(response)