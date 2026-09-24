from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from config.settings import (
    HYBRID_LLM_ROUTING_ENABLED,
    OLLAMA_COMPLEX_MODEL,
    OLLAMA_FAST_MODEL,
    OLLAMA_FORCED_MODEL,
)
from utils.structured_reference import extract_structured_reference


@dataclass(frozen=True)
class ModelRouteDecision:
    """One conservative routing decision for a single user question."""

    route: str
    model_name: Optional[str]
    reason: str
    source_count: int = 0


class ModelRouter:
    """Rule-based LLM router with no extra model call.

    The router deliberately prefers the fast model unless the question clearly
    asks for richer synthesis. Exact Rule/Directive questions can bypass model
    generation entirely when AnswerService has deterministic source-grounded
    extraction for the requested answer type.
    """

    _COMPLEX_PATTERNS = (
        # English explanation / synthesis
        r"\b(?:explain|describe|summari[sz]e|summary|compare|comparison|contrast)\b",
        r"\b(?:difference|differences|differentiate|versus|vs\.?|synthesi[sz]e)\b",
        r"\b(?:across\s+(?:the\s+)?documents|across\s+(?:the\s+)?policies|combined\s+view)\b",
        r"\b(?:overall\s+summary|all\s+relevant\s+documents|multiple\s+documents)\b",
        # Common Tagalog / Filipino wording used in company chat
        r"\b(?:ipaliwanag|paliwanag|ilarawan|ibuod|buod|ikumpara|ihambing)\b",
        r"\b(?:pagkakaiba|pagkakaibang|paghambingin)\b",
    )

    _MULTI_SOURCE_SYNTHESIS_PATTERNS = (
        r"\b(?:both|all|across|combined|overall|together)\b",
        r"\b(?:pareho|lahat|kabuuan|pagsamahin|magkakasama)\b",
    )

    _DETERMINISTIC_FOCUS_PREFIXES = (
        "STRUCTURED STATEMENT:",
        "STRUCTURED SECTION OVERVIEW:",
        "STRUCTURED EXPLANATION:",
        "STRUCTURED DETAIL:",
        "REASON:",
    )

    _STRUCTURED_REFERENCE_PATTERN = re.compile(
        r"\b(?:"
        r"(?:rule|dir(?:ective)?)\s+\d+(?:\.\d+)*"
        r"|(?:section|article|chapter|part)\s+[A-Za-z0-9IVXLCDM]+(?:\.\d+)*"
        r")\b",
        re.IGNORECASE,
    )

    _STRUCTURED_SYNTHESIS_PATTERN = re.compile(
        r"\b(?:compare|comparison|contrast|difference|differences|differentiate|"
        r"versus|vs\.?|summari[sz]e|summary|synthesi[sz]e|ikumpara|ihambing|"
        r"pagkakaiba|pagkakaibang|paghambingin|ibuod|buod)\b",
        re.IGNORECASE,
    )

    def __init__(self):
        self.fast_model = OLLAMA_FAST_MODEL
        self.complex_model = OLLAMA_COMPLEX_MODEL
        self.forced_model = OLLAMA_FORCED_MODEL
        self.enabled = bool(HYBRID_LLM_ROUTING_ENABLED)

    @staticmethod
    def _normalize_question(question: str) -> str:
        return re.sub(r"\s+", " ", str(question or "").strip().lower())

    @staticmethod
    def _source_count(results: Optional[Iterable[Mapping]]) -> int:
        unique = set()

        for item in results or []:
            metadata = item.get("metadata", {}) if isinstance(item, Mapping) else {}
            identity = (
                metadata.get("file_path")
                or metadata.get("file_name")
                or metadata.get("source")
            )
            if identity:
                unique.add(str(identity).strip().casefold())

        return len(unique)

    def question_requires_complex_model(
        self,
        question: str,
        answer_focus: str = "",
        source_count: int = 0,
    ) -> bool:
        """Return True only for clear explanation/synthesis intent."""

        clean = self._normalize_question(question)

        if not clean:
            return False

        if answer_focus.startswith("GROUNDED EXPLANATION:"):
            return True

        if answer_focus.startswith("COMPOUND:"):
            return True

        # Section explanations do not have a deterministic Rule/Directive
        # answer path and benefit from the stronger synthesis model.
        if answer_focus.startswith("STRUCTURED EXPLANATION:"):
            reference = extract_structured_reference(question)
            if reference is not None and reference.is_section_like:
                return True

        if any(re.search(pattern, clean, re.IGNORECASE) for pattern in self._COMPLEX_PATTERNS):
            return True

        if source_count >= 2 and any(
            re.search(pattern, clean, re.IGNORECASE)
            for pattern in self._MULTI_SOURCE_SYNTHESIS_PATTERNS
        ):
            return True

        return False

    def choose(
        self,
        question: str,
        resolved_question: str = "",
        answer_focus: str = "",
        results: Optional[Iterable[Mapping]] = None,
    ) -> ModelRouteDecision:
        """Choose deterministic, fast, complex, or forced generation."""

        source_count = self._source_count(results)
        routing_text = " ".join(
            part
            for part in (question, resolved_question)
            if str(part or "").strip()
        )
        reference = extract_structured_reference(resolved_question or question)
        structured_reference_count = len({
            match.casefold()
            for match in self._STRUCTURED_REFERENCE_PATTERN.findall(routing_text)
        })
        structured_synthesis = bool(
            self._STRUCTURED_SYNTHESIS_PATTERN.search(routing_text)
        )

        deterministic_structured = (
            reference is not None
            and structured_reference_count == 1
            and not structured_synthesis
            and (
                (
                    reference.kind in {"rule", "directive"}
                    and answer_focus.startswith(self._DETERMINISTIC_FOCUS_PREFIXES)
                )
                or (
                    reference.is_section_like
                    and answer_focus.startswith((
                        "STRUCTURED SECTION OVERVIEW:",
                        "STRUCTURED SECTION DETAIL:",
                        "STRUCTURED SECTION TOPICS:",
                    ))
                )
            )
        )

        if deterministic_structured:
            return ModelRouteDecision(
                route="deterministic",
                model_name=None,
                reason=(
                    "Exact structured answer can be derived from the exact "
                    "retrieved block without model generation."
                ),
                source_count=source_count,
            )

        if self.forced_model:
            return ModelRouteDecision(
                route="forced",
                model_name=self.forced_model,
                reason="Process-local DOCUBOT_OLLAMA_MODEL override is active.",
                source_count=source_count,
            )

        if not self.enabled:
            return ModelRouteDecision(
                route="fast",
                model_name=self.fast_model,
                reason="Hybrid routing is disabled; using the configured fast model.",
                source_count=source_count,
            )

        if self.question_requires_complex_model(
            question=routing_text,
            answer_focus=answer_focus,
            source_count=source_count,
        ):
            return ModelRouteDecision(
                route="complex",
                model_name=self.complex_model,
                reason=(
                    "Question clearly requests explanation, comparison, summary, "
                    "compound reasoning, or multi-source synthesis."
                ),
                source_count=source_count,
            )

        return ModelRouteDecision(
            route="fast",
            model_name=self.fast_model,
            reason="Question is direct/simple enough for the low-latency model.",
            source_count=source_count,
        )

    def auxiliary_model_for_question(self, question: str) -> str:
        """Pick a model for query rewrite/translation without a second router call."""

        if self.forced_model:
            return self.forced_model

        if self.question_requires_complex_model(question):
            return self.complex_model

        return self.fast_model
