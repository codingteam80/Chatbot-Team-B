"""Structure-aware PDF extraction for per-file RAG chunking.

The extractor keeps the source document isolated, uses PDF layout/font
information to identify section boundaries, removes repeated page chrome,
and returns logical sections with page-range metadata. It deliberately
falls back to the existing parser when a PDF cannot be processed safely.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from pathlib import Path
from typing import List, Optional, Sequence, Set

import fitz


_RULE_RE = re.compile(
    r"^\s*Rule\s+(?P<id>\d+(?:\.\d+)*)(?:\s*[-:\u2013\u2014]\s*.*)?\s*$",
    re.IGNORECASE,
)
_DIRECTIVE_RE = re.compile(
    r"^\s*(?:Dir|Directive)\s+(?P<id>\d+(?:\.\d+)*)(?:\s*[-:\u2013\u2014]\s*.*)?\s*$",
    re.IGNORECASE,
)
_NAMED_SECTION_RE = re.compile(
    r"^\s*(?:Section|Article|Chapter|Part)\s+"
    r"(?P<id>[A-Za-z0-9IVXLCDM]+(?:\.\d+)*)\b.*$",
    re.IGNORECASE,
)
_NUMBERED_SECTION_RE = re.compile(
    r"^\s*(?P<id>\d+(?:\.\d+){0,4})[.)]?\s+"
    r"(?P<title>[A-Za-z][^\n]{1,100})$"
)
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PDFLine:
    text: str
    page: int
    y0: float
    y1: float
    page_height: float
    max_font_size: float
    bold: bool


@dataclass(frozen=True)
class PDFSection:
    text: str
    page_start: int
    page_end: int
    section_title: str = ""
    section_type: str = "background"
    rule_id: str = ""
    section_id: str = ""


class PDFStructureExtractor:
    """Extract logical sections from one PDF without mixing files."""

    @staticmethod
    def _clean_line_text(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    @classmethod
    def _extract_lines(cls, pdf_path: str) -> List[PDFLine]:
        lines: List[PDFLine] = []

        with fitz.open(str(pdf_path)) as document:
            if document.needs_pass and not document.authenticate(""):
                raise ValueError("PDF is password-protected and requires a password")

            for page_index, page in enumerate(document, start=1):
                page_height = float(page.rect.height or 1.0)
                data = page.get_text("dict", sort=True)

                for block in data.get("blocks", []):
                    for source_line in block.get("lines", []):
                        spans = source_line.get("spans", [])
                        if not spans:
                            continue

                        parts = []
                        max_size = 0.0
                        is_bold = False

                        for span in spans:
                            text = cls._clean_line_text(span.get("text", ""))
                            if not text:
                                continue

                            parts.append(text)
                            size = float(span.get("size", 0.0) or 0.0)
                            max_size = max(max_size, size)

                            font_name = str(span.get("font", "")).lower()
                            flags = int(span.get("flags", 0) or 0)
                            if (flags & 16) or "bold" in font_name:
                                is_bold = True

                        text = cls._clean_line_text(" ".join(parts))
                        if not text:
                            continue

                        bbox = source_line.get("bbox") or (0.0, 0.0, 0.0, 0.0)
                        y0 = float(bbox[1])
                        y1 = float(bbox[3])

                        lines.append(
                            PDFLine(
                                text=text,
                                page=page_index,
                                y0=y0,
                                y1=y1,
                                page_height=page_height,
                                max_font_size=max_size,
                                bold=is_bold,
                            )
                        )

        return lines

    @staticmethod
    def _estimate_body_font_size(lines: Sequence[PDFLine]) -> float:
        sizes = [
            round(line.max_font_size, 1)
            for line in lines
            if line.max_font_size > 0 and len(line.text.split()) >= 3
        ]
        if not sizes:
            return 10.0
        return float(Counter(sizes).most_common(1)[0][0])

    @staticmethod
    def _decoration_key(text: str) -> str:
        normalized = text.casefold()
        normalized = re.sub(r"\bpage\s+\d+(?:\s+of\s+\d+)?\b", "page #", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip(" -|\t")
        return normalized

    @classmethod
    def _detect_repeated_page_decorations(
        cls,
        lines: Sequence[PDFLine],
    ) -> Set[str]:
        pages = {line.page for line in lines}
        if len(pages) < 3:
            return set()

        occurrences = defaultdict(set)

        for line in lines:
            relative_top = line.y0 / max(line.page_height, 1.0)
            relative_bottom = line.y1 / max(line.page_height, 1.0)

            if relative_top > 0.12 and relative_bottom < 0.88:
                continue

            key = cls._decoration_key(line.text)
            if key and len(key) <= 160:
                occurrences[key].add(line.page)

        threshold = max(3, int(math.ceil(len(pages) * 0.5)))
        return {
            key
            for key, page_numbers in occurrences.items()
            if len(page_numbers) >= threshold
        }

    @classmethod
    def _is_page_decoration(cls, line: PDFLine, repeated_keys: Set[str]) -> bool:
        if _PAGE_NUMBER_RE.fullmatch(line.text):
            return True
        return cls._decoration_key(line.text) in repeated_keys

    @staticmethod
    def _is_visually_prominent(line: PDFLine, body_size: float) -> bool:
        short_text = len(line.text) <= 140
        larger_font = line.max_font_size >= body_size * 1.20
        bold_heading = line.bold and line.max_font_size >= body_size * 1.03
        return short_text and (larger_font or bold_heading)

    @classmethod
    def _classify_heading(
        cls,
        line: PDFLine,
        body_size: float,
    ) -> Optional[tuple[str, str, str]]:
        text = line.text.strip()
        if not text or len(text) > 180:
            return None

        match = _RULE_RE.match(text)
        if match and cls._is_visually_prominent(line, body_size):
            return "rule", match.group("id"), ""

        match = _DIRECTIVE_RE.match(text)
        if match and cls._is_visually_prominent(line, body_size):
            return "directive", match.group("id"), ""

        match = _NAMED_SECTION_RE.match(text)
        if match and cls._is_visually_prominent(line, body_size):
            return "section", "", match.group("id")

        match = _NUMBERED_SECTION_RE.match(text)
        if match and cls._is_visually_prominent(line, body_size):
            return "section", "", match.group("id")

        visually_prominent = cls._is_visually_prominent(line, body_size)
        if not visually_prominent:
            return None

        # Avoid treating normal bold sentences as headings.
        if len(text.split()) > 16 or text.endswith((".", "?", "!", ";")):
            return None

        letters = [char for char in text if char.isalpha()]
        uppercase_ratio = (
            sum(char.isupper() for char in letters) / len(letters)
            if letters
            else 0.0
        )

        common_heading_terms = (
            "policy",
            "procedure",
            "purpose",
            "scope",
            "definitions",
            "responsibilities",
            "eligibility",
            "requirements",
            "exceptions",
            "guidelines",
            "approval",
            "benefits",
            "entitlement",
        )

        if uppercase_ratio >= 0.70:
            return "section", "", ""

        if text.casefold().startswith(common_heading_terms):
            return "section", "", ""

        # A visually prominent short title-case line is also a safe
        # generic section boundary for policy/manual PDFs.
        words = [word for word in re.split(r"\s+", text) if word]
        title_like = bool(words) and sum(
            1
            for word in words
            if word[:1].isupper() or word[:1].isdigit()
        ) >= max(1, int(math.ceil(len(words) * 0.6)))

        if title_like and letters:
            return "section", "", ""

        return None

    @classmethod
    def extract(cls, pdf_path: str) -> List[PDFSection]:
        """Return section-aware text units for a single PDF file."""
        path = Path(pdf_path)
        lines = cls._extract_lines(str(path))
        if not lines:
            return []

        repeated_keys = cls._detect_repeated_page_decorations(lines)
        content_lines = [
            line
            for line in lines
            if not cls._is_page_decoration(line, repeated_keys)
        ]
        body_size = cls._estimate_body_font_size(content_lines or lines)

        sections: List[PDFSection] = []
        current_text: List[str] = []
        current_title = ""
        current_type = "background"
        current_rule_id = ""
        current_section_id = ""
        page_start: Optional[int] = None
        page_end: Optional[int] = None

        def flush() -> None:
            nonlocal current_text, current_title, current_type
            nonlocal current_rule_id, current_section_id, page_start, page_end

            text = "\n".join(current_text).strip()
            if text:
                sections.append(
                    PDFSection(
                        text=text,
                        page_start=page_start or 1,
                        page_end=page_end or page_start or 1,
                        section_title=current_title,
                        section_type=current_type,
                        rule_id=current_rule_id,
                        section_id=current_section_id,
                    )
                )

            current_text = []
            current_title = ""
            current_type = "background"
            current_rule_id = ""
            current_section_id = ""
            page_start = None
            page_end = None

        for line in lines:
            if cls._is_page_decoration(line, repeated_keys):
                continue

            heading = cls._classify_heading(line, body_size)

            # A short title-case continuation can be visually prominent even
            # though it is still part of a Rule/Directive requirement sentence
            # (for example a wrapped normative phrase).  Before the mandatory
            # Category row has appeared, do not let a generic unnamed section
            # heading split the authoritative requirement body.  Named semantic
            # roles such as Amplification/Rationale occur after Category and are
            # therefore unaffected.
            if (
                heading is not None
                and current_type in {"rule", "directive"}
                and heading[0] == "section"
                and not heading[2]
                and current_text
                and not any(re.fullmatch(r"(?i)Category", value.strip()) for value in current_text)
            ):
                heading = None

            # For PDFs with no detected structure, keep background text
            # page-local so page metadata remains useful and reliable.
            if (
                heading is None
                and current_type == "background"
                and not current_title
                and current_text
                and page_end is not None
                and line.page != page_end
            ):
                flush()

            if heading is not None:
                flush()
                current_type, current_rule_id, current_section_id = heading
                current_title = line.text
                current_text = [line.text]
                page_start = line.page
                page_end = line.page
                continue

            if page_start is None:
                page_start = line.page
            page_end = line.page
            current_text.append(line.text)

        flush()
        return sections
