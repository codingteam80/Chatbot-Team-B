"""Helpers for reversible filename-safe Unicode marker sequences."""

from __future__ import annotations

import re


_MARKER_RE = re.compile(r"#U([0-9A-Fa-f]{4,6})")


def decode_unicode_markers(value: str) -> str:
    """Decode ``#UXXXX`` / ``#UXXXXXX`` markers without touching normal text.

    Some Windows-safe corpus packages preserve Unicode filename characters as
    marker sequences (for example ``Jos#U00e9 Rizal``). Retrieval/topic logic
    should reason over the human-readable name while file paths stay unchanged.
    """

    text = str(value or "")
    if "#U" not in text and "#u" not in text:
        return text

    def replacement(match: re.Match[str]) -> str:
        try:
            codepoint = int(match.group(1), 16)
            if 0 <= codepoint <= 0x10FFFF:
                return chr(codepoint)
        except (TypeError, ValueError):
            pass
        return match.group(0)

    return _MARKER_RE.sub(replacement, text)
