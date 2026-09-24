"""Utilities for detecting exact structured document identifiers in queries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


_STRUCTURED_REFERENCE_RE = re.compile(
    r"\b(?:"
    r"(?P<rule_kind>rule|dir(?:ective)?)\s*"
    r"(?:no\.?\s*|number\s*)?"
    r"(?P<rule_identifier>\d+(?:\.\d+)*(?:[a-z])?)"
    r"|"
    r"(?P<section_kind>section|article|chapter|part)\b\s*"
    r"(?:no\.?\s*|number\s*)?"
    r"(?P<section_identifier>[A-Za-z0-9IVXLCDM]+(?:\.\d+)*)"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StructuredReference:
    """Normalized exact identifier requested by the user."""

    kind: str
    identifier: str

    @property
    def metadata_id_key(self) -> str:
        return "section_id" if self.is_section_like else "rule_id"

    @property
    def is_section_like(self) -> bool:
        return self.kind in {"section", "article", "chapter", "part"}

    @property
    def section_type(self) -> str:
        if self.kind == "directive":
            return "directive"
        if self.is_section_like:
            # The structure-aware PDF index stores Article/Chapter/Part as
            # section-type records while preserving the original heading text.
            return "section"
        return self.kind

    @property
    def display_name(self) -> str:
        prefix = {
            "rule": "Rule",
            "directive": "Dir",
            "section": "Section",
            "article": "Article",
            "chapter": "Chapter",
            "part": "Part",
        }[self.kind]
        return f"{prefix} {self.identifier}"


def _normalize_identifier(value: str) -> str:
    return (value or "").strip().casefold()


def _is_plausible_section_identifier(value: str) -> bool:
    """Return True for identifier-like Section/Article/Chapter/Part values.

    Section-like references may legitimately use numeric IDs, Roman numerals,
    single letters, or compact alphanumeric IDs such as ``A1``/``A.1``.
    Ordinary prose words (for example ``section for``) must not be treated as
    exact structured identifiers.
    """

    raw = (value or "").strip()
    if not raw:
        return False

    compact = raw.replace(".", "")
    if any(ch.isdigit() for ch in compact):
        return True
    if len(compact) == 1 and compact.isalpha():
        return True
    return bool(re.fullmatch(r"[IVXLCDM]+", compact, flags=re.IGNORECASE))




def extract_structured_references(text: str) -> list[StructuredReference]:
    """Return all plausible exact structured identifiers in text, in order.

    Duplicate references are removed while preserving their first occurrence.
    This supports natural comparison questions such as ``Rule 16.4 vs Rule 16.5``
    without weakening the single-reference API used by older call sites.
    """

    if not text:
        return []

    output: list[StructuredReference] = []
    seen: set[tuple[str, str]] = set()

    for match in _STRUCTURED_REFERENCE_RE.finditer(text):
        raw_kind = (
            match.group("rule_kind")
            or match.group("section_kind")
            or ""
        ).casefold()
        kind = "directive" if raw_kind.startswith("dir") else raw_kind
        identifier = _normalize_identifier(
            match.group("rule_identifier")
            or match.group("section_identifier")
            or ""
        )
        if not identifier:
            continue
        if kind in {"section", "article", "chapter", "part"}:
            if not _is_plausible_section_identifier(identifier):
                continue
        key = (kind, identifier)
        if key in seen:
            continue
        seen.add(key)
        output.append(StructuredReference(kind=kind, identifier=identifier))

    return output

def extract_structured_reference(text: str) -> Optional[StructuredReference]:
    """Return the first plausible exact structured identifier in text."""

    references = extract_structured_references(text)
    return references[0] if references else None
