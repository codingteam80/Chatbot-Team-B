"""Utilities for detecting exact structured document identifiers in queries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


_STRUCTURED_REFERENCE_RE = re.compile(
    r"\b(?P<kind>rule|dir(?:ective)?|section)\s*"
    r"(?:no\.?\s*|number\s*)?"
    r"(?P<identifier>\d+(?:\.\d+)*(?:[a-z])?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StructuredReference:
    """Normalized exact identifier requested by the user."""

    kind: str
    identifier: str

    @property
    def metadata_id_key(self) -> str:
        return "section_id" if self.kind == "section" else "rule_id"

    @property
    def section_type(self) -> str:
        if self.kind == "directive":
            return "directive"
        return self.kind

    @property
    def display_name(self) -> str:
        prefix = {
            "rule": "Rule",
            "directive": "Dir",
            "section": "Section",
        }[self.kind]
        return f"{prefix} {self.identifier}"


def _normalize_identifier(value: str) -> str:
    return (value or "").strip().casefold()


def extract_structured_reference(text: str) -> Optional[StructuredReference]:
    """Return the first exact Rule/Dir/Directive/Section identifier in text."""

    if not text:
        return None

    match = _STRUCTURED_REFERENCE_RE.search(text)
    if not match:
        return None

    raw_kind = match.group("kind").casefold()
    if raw_kind.startswith("dir"):
        kind = "directive"
    else:
        kind = raw_kind

    identifier = _normalize_identifier(match.group("identifier"))
    if not identifier:
        return None

    return StructuredReference(
        kind=kind,
        identifier=identifier,
    )
