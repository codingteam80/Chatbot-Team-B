from __future__ import annotations

import json
import re

from config.prompts import MULTI_QUERY_RETRIEVAL_PROMPT
from config.settings import (
    MULTI_QUERY_MAX_QUERY_CHARS,
    MULTI_QUERY_STRUCTURED_OUTPUT_ENABLED,
    MULTI_QUERY_VARIANT_COUNT,
)


_STRUCTURED_ID_RE = re.compile(
    r"\b(?:rule|directive|dir|section|article|chapter|part)\s+\d+(?:\.\d+)*(?:[a-z])?\b",
    re.IGNORECASE,
)


def _structured_ids(text: str) -> set[str]:
    return {
        re.sub(r"\s+", " ", match.group(0)).strip().casefold()
        for match in _STRUCTURED_ID_RE.finditer(str(text or ""))
    }


def multi_query_json_schema() -> dict:
    """Return the strict Ollama JSON schema for retrieval rewrites."""

    count = int(MULTI_QUERY_VARIANT_COUNT)
    return {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": count,
                "maxItems": count,
            }
        },
        "required": ["queries"],
        "additionalProperties": False,
    }


def _validate_variant_candidates(candidates, original_question: str) -> list[str]:
    """Apply semantic-safety guards to model-produced retrieval rewrites."""

    original = re.sub(r"\s+", " ", str(original_question or "")).strip()
    original_ids = _structured_ids(original)
    seen = {original.casefold()} if original else set()
    variants: list[str] = []

    for candidate in candidates or []:
        if not isinstance(candidate, str):
            continue
        line = candidate.strip('"\'` ')
        line = re.sub(r"\s+", " ", line).strip()
        if not line or len(line) > int(MULTI_QUERY_MAX_QUERY_CHARS):
            continue
        if line.endswith((".", ";")):
            line = line[:-1].strip()
        if not line:
            continue

        # A retrieval rewrite may preserve existing identifiers but must never
        # invent a new Rule/Directive/Section number.
        variant_ids = _structured_ids(line)
        if not variant_ids.issubset(original_ids):
            continue

        # Reject obvious answer-like generations rather than feeding them back
        # into retrieval as if they were user search intent.
        if re.match(
            r"^(?:yes|no|answer|the answer|according to|rule \d+\.?\d* states)\b",
            line,
            re.I,
        ):
            continue

        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        variants.append(line)
        if len(variants) >= int(MULTI_QUERY_VARIANT_COUNT):
            break

    return variants


def parse_multi_query_variants(raw_text: str, original_question: str) -> list[str]:
    """Parse JSON-schema output first, with guarded legacy text compatibility."""

    raw = str(raw_text or "").strip()
    if not raw:
        return []

    # Preferred v6.4.97 path: strict JSON object produced under an Ollama
    # JSON-schema response format. Keep local validation even when the server
    # already enforced the schema so malformed/partial responses fail safely.
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = None

    if isinstance(payload, dict):
        if set(payload.keys()) != {"queries"}:
            return []
        queries = payload.get("queries")
        if not isinstance(queries, list):
            return []
        if len(queries) != int(MULTI_QUERY_VARIANT_COUNT):
            return []
        return _validate_variant_candidates(queries, original_question)

    # Compatibility fallback for older Ollama servers or controlled A/B runs.
    # This preserves the existing line parser and all identifier/answer guards.
    candidates = []
    for raw_line in raw.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw_line).strip()
        line = re.sub(r"^(?:query|variant|alternative)\s*:\s*", "", line, flags=re.I)
        if line:
            candidates.append(line)
    return _validate_variant_candidates(candidates, original_question)


def generate_multi_query_variants(client, question: str) -> list[str]:
    """Generate a small set of intent-preserving search rewrites.

    v6.4.97 prefers strict JSON-schema output at temperature 0. If the local
    Ollama runtime cannot provide structured output, generation falls back to
    the pre-v6.4.97 text path and the same safety parser rather than failing the
    user's RAG request.
    """

    prompt = MULTI_QUERY_RETRIEVAL_PROMPT.format(
        variant_count=int(MULTI_QUERY_VARIANT_COUNT),
        question=str(question or "").strip(),
    )

    raw = ""
    if MULTI_QUERY_STRUCTURED_OUTPUT_ENABLED and hasattr(client, "generate_structured_json"):
        try:
            raw = client.generate_structured_json(prompt, multi_query_json_schema())
        except Exception:
            # Retrieval recall optimization is optional. A structured-generation
            # compatibility problem must not block normal single/multi-query RAG.
            raw = ""

    if not raw:
        raw = client.generate(prompt)

    return parse_multi_query_variants(raw, question)
