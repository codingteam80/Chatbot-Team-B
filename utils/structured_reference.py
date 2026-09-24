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


_GROUPED_RULE_LIST_RE = re.compile(
    r"\b(?P<kind>rules?|directives?|dirs?)\s+"
    r"(?P<identifiers>"
    r"\d+(?:\.\d+)*(?:[a-z])?"
    r"(?:(?:\s*(?:,|and|&|or)\s*|\s+)\d+(?:\.\d+)*(?:[a-z])?)+"
    r")",
    re.IGNORECASE,
)

_MALFORMED_STRUCTURED_REFERENCE_RE = re.compile(
    r"\b(?:MISRA\s+)?(?:Rule|Directive|Dir)\s+"
    r"(?P<identifier>[A-Za-z][A-Za-z0-9_-]*\.[A-Za-z0-9_.-]+)\b",
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
    Besides repeated explicit forms such as ``Rule 10.1 and Rule 10.3``, this
    also supports grouped natural forms such as ``Rules 8.7, 10.1 and 14.3``.
    """

    if not text:
        return []

    positioned: list[tuple[int, int, StructuredReference]] = []
    serial = 0

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
        positioned.append((match.start(), serial, StructuredReference(kind=kind, identifier=identifier)))
        serial += 1

    # Grouped shorthand is common in human questions.  The old singular
    # matcher intentionally did not infer numbers that lacked their own
    # ``Rule`` token, so ``Rules 8.7, 10.1 and 14.3`` silently degraded into
    # broad semantic retrieval.  Parse only an explicitly introduced numeric
    # list and keep the same source kind for every member.
    for match in _GROUPED_RULE_LIST_RE.finditer(text):
        raw_kind = str(match.group("kind") or "").casefold()
        kind = "directive" if raw_kind.startswith("dir") else "rule"
        identifiers = re.findall(
            r"\d+(?:\.\d+)*(?:[a-z])?",
            str(match.group("identifiers") or ""),
            flags=re.IGNORECASE,
        )
        for offset, raw_identifier in enumerate(identifiers):
            identifier = _normalize_identifier(raw_identifier)
            if identifier:
                positioned.append((match.start(), serial + offset, StructuredReference(kind=kind, identifier=identifier)))
        serial += len(identifiers)

    positioned.sort(key=lambda item: (item[0], item[1]))
    output: list[StructuredReference] = []
    seen: set[tuple[str, str]] = set()
    for _position, _serial, reference in positioned:
        key = (reference.kind, reference.identifier)
        if key in seen:
            continue
        seen.add(key)
        output.append(reference)
    return output


def has_malformed_structured_reference(text: str) -> bool:
    """Return True for identifier-looking Rule/Directive tokens that are invalid.

    Keep this deliberately narrow: ordinary prose such as ``what rule applies``
    must not be treated as a malformed exact reference.  Dotted alphabetic
    tokens such as ``Rule ABC.X`` are identifier-shaped and therefore safe to
    reject before semantic retrieval can substitute an unrelated numeric rule.
    """

    if not text:
        return False
    return bool(_MALFORMED_STRUCTURED_REFERENCE_RE.search(str(text)))

def extract_structured_reference(text: str) -> Optional[StructuredReference]:
    """Return the first plausible exact structured identifier in text."""

    references = extract_structured_references(text)
    return references[0] if references else None
