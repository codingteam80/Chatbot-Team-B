"""Phase-6 experimental Rule-aware chunk preparation.

This module is deliberately opt-in.  The certified v6.4.67 production
chunking path remains the default (DOCUBOT_CHUNKING_PROFILE=v3).  v4 only
changes how already structure-extracted PDF sections are represented before
SentenceSplitter enforces the existing size/overlap limits.

Design goals:
- keep every source statement (no source-content deletion);
- inherit the parent Rule/Directive anchor into subordinate sections;
- label semantic roles such as Rationale/Example/Amplification/Exception;
- split an inline ``See also`` tail away from the substantive Example or
  Rationale so cross-reference tokens do not share the same retrieval chunk;
- reset parent scope at a new non-subordinate document section.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Optional

from ingestion.pdf_structure import PDFSection


_SEE_ALSO_LINE_RE = re.compile(r"(?im)^\s*see\s+also\s*$")

# Headings observed in MISRA and intentionally narrow for the first A/B trial.
# Unknown headings are treated as document structure, not silently attached to
# the previous Rule/Directive.
_ROLE_BY_TITLE = {
    "amplification": "amplification",
    "rationale": "rationale",
    "example": "example",
    "examples": "example",
    "exception": "exception",
    "exceptions": "exception",
    "note": "note",
    "notes": "note",
    "see also": "cross_reference",
}


@dataclass(frozen=True)
class ChunkingV4Unit:
    text: str
    page_start: int
    page_end: int
    section_title: str
    section_type: str
    rule_id: str = ""
    section_id: str = ""
    section_role: str = ""
    parent_type: str = ""
    parent_identifier: str = ""
    parent_title: str = ""

    def metadata(self) -> dict:
        data = {
            "chunking_profile": "v4_rule_aware",
            "section_type": self.section_type,
            "section_title": self.section_title,
            "rule_id": self.rule_id,
            "section_id": self.section_id,
            "section_role": self.section_role,
            "parent_type": self.parent_type,
            "parent_identifier": self.parent_identifier,
            "parent_title": self.parent_title,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }
        if self.parent_type == "rule":
            data["parent_rule_id"] = self.parent_identifier
        elif self.parent_type == "directive":
            data["parent_directive_id"] = self.parent_identifier
        return data


class RuleAwareChunkingV4:
    """Transform structure-aware PDF sections into parent-anchored units."""

    @staticmethod
    def _role_for_title(title: str) -> str:
        normalized = re.sub(r"\s+", " ", str(title or "")).strip().casefold()
        return _ROLE_BY_TITLE.get(normalized, "")

    @staticmethod
    def _parent_label(parent_type: str, identifier: str, title: str) -> str:
        if title:
            return title.strip()
        if parent_type == "rule" and identifier:
            return f"Rule {identifier}"
        if parent_type == "directive" and identifier:
            return f"Directive {identifier}"
        return ""

    @classmethod
    def _anchored_text(cls, parent_title: str, text: str) -> str:
        text = str(text or "").strip()
        parent_title = str(parent_title or "").strip()
        if not parent_title or not text:
            return text
        if text.casefold().startswith(parent_title.casefold()):
            return text
        return f"{parent_title}\n{text}"

    @staticmethod
    def _split_see_also(text: str) -> tuple[str, str]:
        """Return substantive text and an optional inline See-also tail."""
        text = str(text or "").strip()
        match = _SEE_ALSO_LINE_RE.search(text)
        if not match:
            return text, ""
        before = text[: match.start()].strip()
        after = text[match.end() :].strip()
        cross = "See also"
        if after:
            cross = f"{cross}\n{after}"
        return before, cross

    @classmethod
    def expand(cls, sections: Iterable[PDFSection]) -> List[ChunkingV4Unit]:
        units: List[ChunkingV4Unit] = []
        parent_type = ""
        parent_identifier = ""
        parent_title = ""

        for section in sections:
            section_type = str(section.section_type or "").strip().casefold()
            title = str(section.section_title or "").strip()
            role = cls._role_for_title(title)

            if section_type in {"rule", "directive"}:
                parent_type = section_type
                parent_identifier = str(section.rule_id or section.section_id or "").strip()
                parent_title = cls._parent_label(parent_type, parent_identifier, title)
                units.append(
                    ChunkingV4Unit(
                        text=str(section.text or "").strip(),
                        page_start=section.page_start,
                        page_end=section.page_end,
                        section_title=title,
                        section_type=section.section_type,
                        rule_id=section.rule_id,
                        section_id=section.section_id,
                        section_role="requirement",
                        parent_type=parent_type,
                        parent_identifier=parent_identifier,
                        parent_title=parent_title,
                    )
                )
                continue

            # Rationale/Example/etc. directly after a Rule/Directive inherit that
            # parent.  Any other detected document section closes the parent
            # scope so a chapter heading never becomes part of the last rule.
            if not role:
                parent_type = ""
                parent_identifier = ""
                parent_title = ""
                units.append(
                    ChunkingV4Unit(
                        text=str(section.text or "").strip(),
                        page_start=section.page_start,
                        page_end=section.page_end,
                        section_title=title,
                        section_type=section.section_type,
                        rule_id=section.rule_id,
                        section_id=section.section_id,
                        section_role="document_section",
                    )
                )
                continue

            main_text, cross_text = cls._split_see_also(section.text)
            active_parent_title = parent_title if parent_type else ""
            if main_text:
                units.append(
                    ChunkingV4Unit(
                        text=cls._anchored_text(active_parent_title, main_text),
                        page_start=section.page_start,
                        page_end=section.page_end,
                        section_title=title,
                        section_type=section.section_type,
                        rule_id=section.rule_id,
                        section_id=section.section_id,
                        section_role=role,
                        parent_type=parent_type,
                        parent_identifier=parent_identifier,
                        parent_title=active_parent_title,
                    )
                )

            if cross_text:
                units.append(
                    ChunkingV4Unit(
                        text=cls._anchored_text(active_parent_title, cross_text),
                        page_start=section.page_start,
                        page_end=section.page_end,
                        section_title="See also",
                        section_type=section.section_type,
                        rule_id=section.rule_id,
                        section_id=section.section_id,
                        section_role="cross_reference",
                        parent_type=parent_type,
                        parent_identifier=parent_identifier,
                        parent_title=active_parent_title,
                    )
                )

        return units
