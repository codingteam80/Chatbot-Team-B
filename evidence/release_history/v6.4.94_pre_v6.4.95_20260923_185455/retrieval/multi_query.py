from __future__ import annotations

import re

from config.prompts import MULTI_QUERY_RETRIEVAL_PROMPT
from config.settings import MULTI_QUERY_MAX_QUERY_CHARS, MULTI_QUERY_VARIANT_COUNT


def parse_multi_query_variants(raw_text: str, original_question: str) -> list[str]:
    """Parse/validate alternative retrieval queries without adding knowledge."""

    original = re.sub(r"\s+", " ", str(original_question or "")).strip()
    seen = {original.casefold()} if original else set()
    variants = []

    for raw_line in str(raw_text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw_line).strip()
        line = line.strip('"\'` ')
        line = re.sub(r"\s+", " ", line).strip()
        if not line or len(line) > int(MULTI_QUERY_MAX_QUERY_CHARS):
            continue
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        variants.append(line)
        if len(variants) >= int(MULTI_QUERY_VARIANT_COUNT):
            break

    return variants


def generate_multi_query_variants(client, question: str) -> list[str]:
    """Generate controlled search variants with an injected local LLM client.

    This helper is experiment-ready but is not enabled in the production
    retrieval path by default.  RAGAS/Golden evaluation decides whether the
    MULTI_QUERY_RETRIEVAL_ENABLED flag should later be promoted.
    """

    prompt = MULTI_QUERY_RETRIEVAL_PROMPT.format(
        variant_count=int(MULTI_QUERY_VARIANT_COUNT),
        question=str(question or "").strip(),
    )
    raw = client.generate(prompt)
    return parse_multi_query_variants(raw, question)
