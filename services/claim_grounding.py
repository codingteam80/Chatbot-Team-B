from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class GroundingCheck:
    ok: bool
    reasons: tuple[str, ...] = ()


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text or ""))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _available_structured_references(results: Iterable[Mapping]) -> set[str]:
    refs = set()
    for item in results or []:
        if not isinstance(item, Mapping):
            continue
        meta = item.get("metadata", {}) or {}
        section_type = str(meta.get("section_type") or "").strip().casefold()
        if section_type in {"rule", "directive", "section", "article", "chapter", "part"}:
            key = "rule_id" if section_type in {"rule", "directive"} else "section_id"
            identifier = str(meta.get(key) or "").strip().casefold()
            if identifier:
                refs.add(f"{section_type}:{identifier}")
    return refs


def _structured_references(text: str) -> set[str]:
    found = set()
    pattern = re.compile(
        r"\b(?P<kind>Rule|Directive|Dir\.?|Section|Article|Chapter|Part)\s+"
        r"(?P<id>\d+(?:\.\d+)*)\b",
        re.IGNORECASE,
    )
    for match in pattern.finditer(str(text or "")):
        kind = match.group("kind").casefold().rstrip(".")
        if kind == "dir":
            kind = "directive"
        found.add(f"{kind}:{match.group('id').casefold()}")
    return found


def _numeric_tokens(text: str) -> set[str]:
    # Remove Markdown ordered-list markers so formatting "1." / "2." does not
    # look like a factual claim introduced by the model.
    clean = re.sub(r"(?m)^\s*\d+[.)]\s+", "", str(text or ""))
    return set(
        re.findall(
            r"(?<![A-Za-z0-9_])\d+(?:\.\d+)*(?:%|ms|s|sec|secs|seconds?|"
            r"minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|MB|GB|KB)?",
            clean,
            flags=re.IGNORECASE,
        )
    )


def _proper_name_candidates(text: str) -> set[str]:
    candidates = set()
    generic = {
        "the", "this", "that", "these", "those", "yes", "no", "needs",
        "assessment", "conclusion", "applicable", "required", "potential",
        "source", "company", "knowledge", "information", "rule", "directive",
        "section", "article", "chapter", "part", "misra", "c", "cpp",
    }
    pattern = re.compile(
        r"\b[A-Z][A-Za-z0-9_+#./-]*(?:\s+[A-Z][A-Za-z0-9_+#./-]*){1,4}\b"
    )
    for match in pattern.finditer(str(text or "")):
        value = match.group(0).strip()
        tokens = [token.casefold().strip(".,:;()[]{}") for token in value.split()]
        meaningful = [token for token in tokens if token and token not in generic]
        if len(meaningful) >= 2:
            candidates.add(value)
    return candidates


def _acronym_candidates(text: str) -> set[str]:
    return {
        value
        for value in re.findall(r"\b[A-Z][A-Z0-9_-]{2,}\b", str(text or ""))
        if value not in {"MISRA", "NOTE", "TODO"}
    }


def validate_generated_claims(
    *,
    answer: str,
    context: str,
    question: str = "",
    results=(),
) -> GroundingCheck:
    """Conservative deterministic hallucination guard for generated answers.

    It does not attempt semantic fact checking.  It rejects only high-confidence
    additions that should never appear without evidence: unseen numeric values,
    structured identifiers, multi-token proper names, or acronyms.  This keeps
    the guard useful without penalizing ordinary paraphrase.
    """

    if not answer or not context:
        return GroundingCheck(ok=True)

    evidence = _normalize(context + "\n" + question)
    reasons = []

    available_refs = _available_structured_references(results)
    cited_refs = _structured_references(answer)
    unsupported_refs = sorted(
        ref for ref in cited_refs
        if ref not in available_refs and _normalize(ref.split(":", 1)[1]) not in evidence
    )
    if unsupported_refs:
        reasons.append("unsupported structured reference(s): " + ", ".join(unsupported_refs))

    evidence_numbers = _numeric_tokens(context + "\n" + question)
    answer_numbers = _numeric_tokens(answer)
    unsupported_numbers = sorted(answer_numbers - evidence_numbers)
    if unsupported_numbers:
        reasons.append("unsupported numeric value(s): " + ", ".join(unsupported_numbers))

    for name in sorted(_proper_name_candidates(answer)):
        if _normalize(name) not in evidence:
            reasons.append(f"unsupported proper name/entity: {name}")

    evidence_acronyms = _acronym_candidates(context + "\n" + question)
    for acronym in sorted(_acronym_candidates(answer) - evidence_acronyms):
        reasons.append(f"unsupported acronym: {acronym}")

    return GroundingCheck(ok=not reasons, reasons=tuple(reasons))
