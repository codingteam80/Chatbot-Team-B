from __future__ import annotations

import math
from typing import Iterable

import requests

from config.settings import (
    DEFAULT_NORMALIZE_EMBEDDINGS,
    EMBED_MODEL_NAME,
    EMBED_OLLAMA_KEEP_ALIVE,
    EMBED_OLLAMA_TIMEOUT,
    EMBED_OLLAMA_CONNECT_TIMEOUT,
    EMBED_OLLAMA_URL,
)


class OllamaEmbeddingModel:
    """Small Ollama /api/embed adapter for Qwen3 production embeddings.

    The class exposes the long-standing embedding methods used by indexing and
    retrieval while the finalized runtime has a single embedding backend.
    """

    def __init__(self):
        self.model_name = EMBED_MODEL_NAME
        self.base_url = EMBED_OLLAMA_URL.rstrip("/")
        self.timeout = float(EMBED_OLLAMA_TIMEOUT)
        self.connect_timeout = float(EMBED_OLLAMA_CONNECT_TIMEOUT)
        self.keep_alive = EMBED_OLLAMA_KEEP_ALIVE
        self.normalize = bool(DEFAULT_NORMALIZE_EMBEDDINGS)

    @staticmethod
    def _normalize(vector):
        values = [float(value) for value in vector]
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 0:
            return values
        return [value / norm for value in values]

    def _post(self, url: str, *, json_payload: dict):
        """POST to local Ollama with a short connect timeout and one safe retry.

        Only connection-establishment failures are retried. Read/generation
        timeouts are not doubled, so a slow model cannot turn one timeout into
        two long waits.
        """

        import time

        last_error = None
        for attempt in range(2):
            try:
                return requests.post(
                    url,
                    json=json_payload,
                    timeout=(self.connect_timeout, self.timeout),
                )
            except requests.ConnectionError as error:
                last_error = error
                if attempt == 0:
                    time.sleep(0.35)
                    continue
                raise
        raise last_error  # pragma: no cover

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []

        payload = {
            "model": self.model_name,
            "input": inputs,
            "keep_alive": self.keep_alive,
        }

        try:
            response = self._post(
                self.base_url + "/api/embed",
                json_payload=payload,
            )
            if response.status_code == 404:
                # Compatibility with older Ollama builds.  The legacy endpoint
                # is single-input only, so preserve order with one request per
                # text rather than silently dropping batch items.
                legacy_vectors = []
                for value in inputs:
                    legacy = self._post(
                        self.base_url + "/api/embeddings",
                        json_payload={"model": self.model_name, "prompt": value},
                    )
                    legacy.raise_for_status()
                    legacy_body = legacy.json()
                    vector = legacy_body.get("embedding")
                    if not isinstance(vector, list) or not vector:
                        raise RuntimeError(
                            "Legacy Ollama embedding endpoint returned no vector."
                        )
                    legacy_vectors.append(vector)
                embeddings = legacy_vectors
            else:
                response.raise_for_status()
                try:
                    body = response.json()
                except ValueError as error:
                    raise RuntimeError(
                        "Ollama embedding endpoint returned invalid JSON."
                    ) from error
                embeddings = body.get("embeddings")
        except requests.RequestException as error:
            raise RuntimeError(
                "Ollama embedding request failed. Ensure Ollama is running and "
                f"the model '{self.model_name}' is installed. Detail: {error}"
            ) from error

        if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
            raise RuntimeError(
                "Ollama embedding response did not contain one vector per input."
            )

        output = []
        for vector in embeddings:
            if not isinstance(vector, list) or not vector:
                raise RuntimeError("Ollama embedding response contained an empty vector.")
            values = [float(value) for value in vector]
            if self.normalize:
                values = self._normalize(values)
            output.append(values)
        return output

    QUERY_INSTRUCTION = (
        "Instruct: Given a technical/software-engineering query, retrieve "
        "relevant passages from internal standards, specifications, manuals, "
        "and engineering documents that directly answer the query.\nQuery: "
    )

    def get_query_embedding(self, text: str):
        """Embed one retrieval query using Qwen3's recommended instruction style."""
        return self._embed([self.QUERY_INSTRUCTION + str(text or "")])[0]

    def get_text_embedding(self, text: str):
        """Embed one document/passage without a query instruction."""
        return self._embed([str(text or "")])[0]

    def get_text_embedding_batch(self, texts: Iterable[str]):
        """Embed document/passages in order; documents need no instruction."""
        return self._embed([str(text or "") for text in texts])
