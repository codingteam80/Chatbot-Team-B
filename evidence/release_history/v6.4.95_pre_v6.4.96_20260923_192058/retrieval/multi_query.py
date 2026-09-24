from __future__ import annotations

import re

from config.prompts import MULTI_QUERY_RETRIEVAL_PROMPT
from config.settings import MULTI_QUERY_MAX_QUERY_CHARS, MULTI_QUERY_VARIANT_COUNT


_STRUCTURED_ID_RE = re.compile(
    r"\b(?:rule|directive|dir|section|article|chapter|part)\s+\d+(?:\.\d+)*(?:[a-z])?\b",
    re.IGNORECASE,
)


def _structured_ids(text: str) -> set[str]:
    return {
        re.sub(r"\s+", " ", match.group(0)).strip().casefold()
        for match in _STRUCTURED_ID_RE.finditer(str(text or ""))
    }


def parse_multi_query_variants(raw_text: str, original_question: str) -> list[str]:
    """Parse controlled retrieval variants without allowing identifier drift."""

    original = re.sub(r"\s+", " ", str(original_question or "")).strip()
    original_ids = _structured_ids(original)
    seen = {original.casefold()} if original else set()
    variants: list[str] = []

    for raw_line in str(raw_text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw_line).strip()
        line = re.sub(r"^(?:query|variant|alternative)\s*:\s*", "", line, flags=re.I)
        line = line.strip('"\'` ')
        line = re.sub(r"\s+", " ", line).strip()
        if not line or len(line) > int(MULTI_QUERY_MAX_QUERY_CHARS):
            continue
        if line.endswith(('.', ';')):
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
        if re.match(r"^(?:yes|no|answer|the answer|according to|rule \d+\.?\d* states)\b", line, re.I):
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
    """Generate a small set of intent-preserving search rewrites."""

    prompt = MULTI_QUERY_RETRIEVAL_PROMPT.format(
        variant_count=int(MULTI_QUERY_VARIANT_COUNT),
        question=str(question or "").strip(),
    )
    raw = client.generate(prompt)
    return parse_multi_query_variants(raw, question)
