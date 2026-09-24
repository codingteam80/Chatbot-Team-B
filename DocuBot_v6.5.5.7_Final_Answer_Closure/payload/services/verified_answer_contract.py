"""Final grounded-answer contract for verified structured references.

This module is deliberately deterministic and corpus-driven.  It never maps a
question phrase to a Rule/Directive number.  It only uses identifiers that are
already present in the accepted retrieval result metadata.

The contract closes two rare generation-quality gaps:
1. an answer can preserve the correct yes/no polarity but omit the sole
   verified Rule/Directive identifier (for example, returning only ``No.``);
2. a local model can occasionally switch to an unrelated Unicode script even
   though the user's question is Latin-script.

When either condition occurs, the finalizer fails closed to a short answer
built only from the already verified structured reference and the model's
accepted polarity.  Normal fluent answers are left unchanged.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Mapping

_REF_RE = re.compile(
    r"\b(?P<kind>Rule|Directive|Dir\.?)\s+(?P<identifier>\d+(?:\.\d+)+)\b",
    re.IGNORECASE,
)

_TAGALOG_CUES = {
    "ano", "anong", "alin", "aling", "kung", "may", "ang", "ng", "mga",
    "sa", "ba", "dapat", "pwede", "puwede", "kapag", "tapos", "bago",
    "gamit", "ginamit", "relevant", "applicable",
}


def _canonical_label(kind: str, identifier: str) -> str:
    kind_clean = str(kind or "").strip().casefold().rstrip(".")
    label = "Directive" if kind_clean.startswith("dir") else "Rule"
    return f"{label} {str(identifier or '').strip()}".strip()


def available_structured_references(results: Iterable[Mapping]) -> tuple[str, ...]:
    """Return verified Rule/Directive labels from accepted result metadata only."""

    labels: list[str] = []
    seen: set[str] = set()
    for item in results or []:
        if not isinstance(item, Mapping):
            continue
        metadata = item.get("metadata", {}) or {}
        section_type = str(metadata.get("section_type", "") or "").strip().casefold()
        if section_type not in {"rule", "directive"}:
            continue
        if section_type == "directive":
            identifier = str(
                metadata.get("directive_id", "")
                or metadata.get("rule_id", "")
                or metadata.get("section_id", "")
                or ""
            ).strip()
            kind = "Directive"
        else:
            identifier = str(
                metadata.get("rule_id", "")
                or metadata.get("section_id", "")
                or ""
            ).strip()
            kind = "Rule"
        if not identifier:
            continue
        label = _canonical_label(kind, identifier)
        key = label.casefold()
        if key not in seen:
            seen.add(key)
            labels.append(label)
    return tuple(labels)


def cited_structured_references(answer: str) -> tuple[str, ...]:
    labels: list[str] = []
    seen: set[str] = set()
    for match in _REF_RE.finditer(str(answer or "")):
        label = _canonical_label(match.group("kind"), match.group("identifier"))
        key = label.casefold()
        if key not in seen:
            seen.add(key)
            labels.append(label)
    return tuple(labels)


def _script_families(text: str) -> set[str]:
    """Return broad non-Latin script families used by alphabetic characters."""

    families: set[str] = set()
    for ch in str(text or ""):
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if "LATIN" in name or "COMBINING" in name:
            continue
        if any(marker in name for marker in ("CJK", "IDEOGRAPH")):
            families.add("CJK")
        elif "HIRAGANA" in name:
            families.add("HIRAGANA")
        elif "KATAKANA" in name:
            families.add("KATAKANA")
        elif "HANGUL" in name:
            families.add("HANGUL")
        elif "CYRILLIC" in name:
            families.add("CYRILLIC")
        elif "ARABIC" in name:
            families.add("ARABIC")
        elif "HEBREW" in name:
            families.add("HEBREW")
        elif "DEVANAGARI" in name:
            families.add("DEVANAGARI")
        elif "THAI" in name:
            families.add("THAI")
        else:
            # Unknown non-Latin alphabetic scripts are still tracked so a model
            # cannot silently drift to an unrelated writing system.
            families.add(name.split(" ", 1)[0])
    return families


def unexpected_script_families(question: str, answer: str) -> tuple[str, ...]:
    question_scripts = _script_families(question)
    answer_scripts = _script_families(answer)
    return tuple(sorted(answer_scripts - question_scripts))


def _looks_tagalog_or_taglish(question: str) -> bool:
    words = re.findall(r"[a-zA-Z]+", str(question or "").casefold())
    if not words:
        return False
    hits = sum(1 for word in set(words) if word in _TAGALOG_CUES)
    return hits >= 2 or (words[0] in {"ano", "anong", "alin", "kung", "may", "paano", "bakit"})


def _polarity_prefix(answer: str) -> str:
    clean = str(answer or "").lstrip()
    match = re.match(r"(?i)^(yes|no|needs\s+more\s+context)\b", clean)
    if not match:
        return ""
    raw = match.group(1).casefold()
    if raw == "yes":
        return "Yes."
    if raw == "no":
        return "No."
    return "Needs more context."


def _question_requests_structured_reference(question: str) -> bool:
    clean = re.sub(r"\s+", " ", str(question or "").casefold())
    if re.search(r"\b(?:rule|rules|directive|directives|requirement|requirements)\b", clean):
        return True
    if "misra" in clean and re.search(
        r"\b(?:which|what|anong|ano|alin|aling|applicable|relevant|concerned|covers|addresses)\b",
        clean,
    ):
        return True
    return False


def _minimal_grounded_fallback(
    *,
    reference: str,
    question: str,
    answer: str,
    yes_no_intent: bool,
) -> str:
    taglish = _looks_tagalog_or_taglish(question)
    polarity = _polarity_prefix(answer) if yes_no_intent else ""

    if taglish:
        if polarity:
            return f"{polarity} {reference} ang applicable MISRA requirement."
        return f"Ang applicable MISRA requirement ay {reference}."

    if polarity:
        return f"{polarity} {reference} is the applicable MISRA requirement."
    return f"The applicable MISRA requirement is {reference}."


def enforce_verified_answer_contract(
    *,
    answer: str,
    question: str,
    results: Iterable[Mapping],
    yes_no_intent: bool = False,
    require_reference: bool = False,
) -> tuple[str, dict]:
    """Apply a conservative final contract over already verified retrieval.

    Returns ``(answer, changes)``.  ``changes`` is empty when no correction was
    required.  The function never invents an identifier: it acts only when
    exactly one Rule/Directive is proven by result metadata.
    """

    original = str(answer or "").strip()
    if not original:
        return original, {}

    references = available_structured_references(results)
    if len(references) != 1:
        return original, {}
    reference = references[0]

    script_drift = unexpected_script_families(question, original)
    cited = {value.casefold() for value in cited_structured_references(original)}
    missing_reference = reference.casefold() not in cited
    should_require_reference = bool(
        require_reference
        or yes_no_intent
        or _question_requests_structured_reference(question)
    )

    if script_drift:
        corrected = _minimal_grounded_fallback(
            reference=reference,
            question=question,
            answer=original,
            yes_no_intent=yes_no_intent,
        )
        return corrected, {
            "applied": True,
            "reason": "unexpected_language_script",
            "unexpected_scripts": list(script_drift),
            "verified_reference": reference,
            "reference_was_missing": missing_reference,
        }

    if missing_reference and should_require_reference:
        clean = original.strip()
        polarity = _polarity_prefix(clean) if yes_no_intent else ""
        # Very short yes/no outputs are safer to replace with a complete grounded
        # sentence than to append a dangling reference token.
        substantive = re.sub(
            r"(?i)^(?:yes|no|needs\s+more\s+context)\s*[.!,:;\-]*\s*",
            "",
            clean,
        ).strip()
        if polarity and not substantive:
            corrected = _minimal_grounded_fallback(
                reference=reference,
                question=question,
                answer=clean,
                yes_no_intent=True,
            )
        elif polarity:
            # Keep the model's grounded explanation but make the verified source
            # identity explicit immediately after the accepted polarity.
            rest = re.sub(
                r"(?i)^(?:yes|no|needs\s+more\s+context)\s*[.!,:;\-]*\s*",
                "",
                clean,
            ).strip()
            corrected = f"{polarity} {reference}: {rest}" if rest else _minimal_grounded_fallback(
                reference=reference,
                question=question,
                answer=clean,
                yes_no_intent=True,
            )
        else:
            corrected = f"{reference}: {clean}"

        return corrected.strip(), {
            "applied": True,
            "reason": "verified_reference_missing_from_generated_answer",
            "verified_reference": reference,
        }

    return original, {}
