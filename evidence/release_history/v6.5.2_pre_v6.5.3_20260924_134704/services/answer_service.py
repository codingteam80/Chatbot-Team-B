from pathlib import Path
import json
import re
import time
import unicodedata

from llm.ollama_client import OllamaClient
from llm.model_router import ModelRouteDecision, ModelRouter
from services.query_service import QueryService

from config.prompts import (
    SYSTEM_PROMPT,
    ANSWER_TEMPLATE,
    NO_RESULT_MESSAGE,
    REWRITE_QUERY_PROMPT,
    MULTILINGUAL_RETRIEVAL_QUERY_PROMPT,
    MISRA_COMPLIANCE_TEMPLATE,
)

from config.settings import (
    DEBUG_MODE,
    ENABLE_MULTILINGUAL_RETRIEVAL,
    MULTILINGUAL_RETRY_ON_EMPTY,
    MULTILINGUAL_QUERY_MAX_CHARS,
    OLLAMA_FAST_MODEL,
    OLLAMA_FORCED_MODEL,
    LLM_CERTIFICATION_FORCE_GENERATION,
    GENERATION_USE_FULL_ACCEPTED_CONTEXT,
    KNOWLEDGE_PROFILE,
)
from chat.chat_manager import ChatManager
from chat.query_normalizer import QueryNormalizer
from chat.conversation_resolver import ConversationResolver
from chat.query_enricher import QueryEnricher
from services.misra_compliance import MisraComplianceMode
from services.claim_grounding import validate_generated_claims
from qa.evidence_logger import evidence_logger
from utils.structured_reference import (
    StructuredReference,
    extract_structured_reference,
    extract_structured_references,
    has_malformed_structured_reference,
)
from utils.unicode_markers import decode_unicode_markers


class AnswerService:

    def __init__(self):

        # Retrieve relevant company knowledge
        self.query_service = (
            QueryService()
        )

        # Hybrid LLM routing is rule-based and requires no extra model call.
        self.model_router = ModelRouter()

        # Keep the legacy/auxiliary client lazy. With no forced A/B override,
        # auxiliary simple tasks use the fast model. No Ollama model is loaded
        # merely by constructing AnswerService.
        default_model = OLLAMA_FORCED_MODEL or OLLAMA_FAST_MODEL
        self.llm = OllamaClient(default_model)
        self._llm_clients = {
            self.llm.model_name: self.llm
        }

        # Normalize retrieval queries
        self.query_normalizer = (
            QueryNormalizer()
        )

        # Resolve follow-up questions
        self.conversation_resolver = (
            ConversationResolver()
        )

        # Expand simple topic queries
        self.query_enricher = (
            QueryEnricher()
        )

        # Cache canonical English retrieval queries so repeated
        # QA runs do not require another translation call.
        self._multilingual_query_cache = {}

    def _get_llm_client(self, model_name: str | None = None):

        """Return one lazy per-model client without loading unused models."""

        resolved_model = str(
            model_name
            or OLLAMA_FORCED_MODEL
            or OLLAMA_FAST_MODEL
        ).strip()

        clients = getattr(self, "_llm_clients", None)

        if clients is None:
            clients = {}
            self._llm_clients = clients

        client = clients.get(resolved_model)

        if client is None:
            client = OllamaClient(resolved_model)
            clients[resolved_model] = client

        return client

    def _auxiliary_llm_for_question(self, question: str):

        """Use the likely final-route model for translation/rewrite helpers."""

        router = getattr(self, "model_router", None)

        if router is None:
            router = ModelRouter()
            self.model_router = router

        model_name = router.auxiliary_model_for_question(
            question
        )

        return self._get_llm_client(model_name)

    def _generate_with_latency(
        self,
        client,
        prompt: str,
        purpose: str,
    ):

        """Generate once and record model-call latency for QA diagnostics."""

        started = time.perf_counter()

        try:
            return client.generate(prompt)
        finally:
            elapsed = time.perf_counter() - started
            evidence_logger.record_event(
                event_name="LLM CALL LATENCY",
                status="RECORDED",
                details={
                    "purpose": purpose,
                    "model": getattr(client, "model_name", "Unknown"),
                    "seconds": round(elapsed, 4),
                    "context_window": getattr(client, "context_window", ""),
                    "keep_alive": getattr(client, "keep_alive", ""),
                },
            )

    def _generate_for_route(
        self,
        prompt: str,
        decision: ModelRouteDecision,
    ):

        """Generate with the routed model and safely degrade complex->fast.

        A complex-model failure must not break the whole chatbot when the fast
        local model is still available. Explicit forced-model diagnostics do
        not silently switch models because that would invalidate the test.
        """

        if not decision.model_name:
            raise ValueError("A generative LLM route requires a model name.")

        client = self._get_llm_client(
            decision.model_name
        )

        try:
            return self._generate_with_latency(
                client,
                prompt,
                purpose="answer_generation",
            ), client

        except Exception as error:
            router = getattr(self, "model_router", None)

            can_fallback = (
                decision.route == "complex"
                and router is not None
                and not router.forced_model
                and decision.model_name != router.fast_model
            )

            if not can_fallback:
                raise

            evidence_logger.record_event(
                event_name="LLM ROUTE FALLBACK",
                status="COMPLEX MODEL FAILED; USING FAST MODEL",
                details={
                    "requested_model": decision.model_name,
                    "fallback_model": router.fast_model,
                    "error_type": type(error).__name__,
                },
            )

            fallback_client = self._get_llm_client(
                router.fast_model
            )

            return self._generate_with_latency(
                fallback_client,
                prompt,
                purpose="answer_generation_fast_fallback",
            ), fallback_client

    def _schedule_post_complex_fast_recovery(self, generation_client) -> None:
        """Restore the fast Ollama model after all complex answer work is done."""

        if generation_client is None:
            return

        try:
            from runtime.model_residency import (
                schedule_fast_model_recovery_after_complex,
            )

            result = schedule_fast_model_recovery_after_complex(
                getattr(generation_client, "model_name", "")
            )
            if result.get("scheduled"):
                evidence_logger.record_event(
                    event_name="FAST MODEL RESIDENCY RECOVERY",
                    status="SCHEDULED",
                    details={
                        "after_model": getattr(
                            generation_client, "model_name", "Unknown"
                        )
                    },
                )
        except Exception as error:
            # Runtime optimization must never affect answer correctness.
            evidence_logger.record_event(
                event_name="FAST MODEL RESIDENCY RECOVERY",
                status="SKIPPED",
                details={"error_type": type(error).__name__},
            )

    def _semantic_misra_relation_answer(
        self,
        question: str,
        results,
    ):
        """Resolve one semantic Rule Yes/No relation with a compact grounded call.

        Rule selection has already been performed by the corpus-derived semantic
        resolver and independently verified against the authoritative BM25 source.
        This call decides only the requested proposition polarity.  It replaces
        brittle phrase-specific polarity code and avoids a full-context answer
        generation pass when a single exact requirement is enough.
        """

        if not MisraComplianceMode.yes_no_intent(question):
            return ""
        candidates = [
            item for item in (results or [])
            if isinstance(item, dict)
            and item.get("_misra_semantic_resolver") is True
            and MisraComplianceMode._is_citable_rule_body_record(item)
        ]
        if len(candidates) != 1:
            return ""

        item = candidates[0]
        metadata = item.get("metadata", {}) or {}
        identifier = str(
            metadata.get("directive_id", "")
            or metadata.get("rule_id", "")
            or ""
        ).strip()
        if not identifier:
            return ""
        kind = "Directive" if str(metadata.get("section_type", "") or "").casefold() == "directive" else "Rule"
        reference = f"{kind} {identifier}"
        statement = MisraComplianceMode._source_rule_statement(
            str(item.get("text", "") or "")
        )
        if not statement:
            return ""

        schema = {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["yes", "no", "needs_context"],
                }
            },
            "required": ["verdict"],
            "additionalProperties": False,
        }
        prompt = (
            "You are a strict source-grounded standards relation checker.\n"
            "Use ONLY the exact SOURCE REQUIREMENT below. Do not use outside knowledge.\n"
            "Interpret YES as: the proposition asked by the user is permitted/satisfied by the requirement.\n"
            "Interpret NO as: the proposition asked by the user conflicts with or is disallowed by the requirement.\n"
            "Use NEEDS_CONTEXT only when the exact requirement does not determine the proposition.\n"
            "Return only the requested JSON object. Do not explain your reasoning.\n\n"
            f"SOURCE REFERENCE: {reference}\n"
            f"SOURCE REQUIREMENT: {statement}\n"
            f"USER QUESTION: {str(question or '').strip()}"
        )

        client = self._get_llm_client(OLLAMA_FAST_MODEL)
        started = time.perf_counter()
        try:
            raw = client.generate_structured_json(prompt, schema)
        except Exception as error:
            evidence_logger.record_event(
                event_name="MISRA SEMANTIC RELATION VERIFIER",
                status="FALLBACK TO NORMAL GROUNDED GENERATION",
                details={
                    "reference": reference,
                    "error_type": type(error).__name__,
                    "seconds": round(time.perf_counter() - started, 4),
                },
            )
            return ""

        evidence_logger.record_event(
            event_name="LLM CALL LATENCY",
            status="RECORDED",
            details={
                "purpose": "misra_semantic_relation_verifier",
                "model": getattr(client, "model_name", "Unknown"),
                "seconds": round(time.perf_counter() - started, 4),
                "context_window": getattr(client, "context_window", ""),
                "keep_alive": getattr(client, "keep_alive", ""),
            },
        )
        try:
            payload = json.loads(str(raw or ""))
        except Exception:
            return ""
        verdict = str(payload.get("verdict", "") or "").strip().casefold()
        if verdict not in {"yes", "no", "needs_context"}:
            return ""

        statement_display = statement.rstrip(" .")
        if statement_display:
            statement_display = statement_display[:1].lower() + statement_display[1:]
        if verdict == "yes":
            answer = f"Yes. {reference} states that {statement_display}."
        elif verdict == "no":
            answer = f"No. {reference} states that {statement_display}."
        else:
            answer = f"Needs more context. {reference} states that {statement_display}."

        if not MisraComplianceMode.references_are_grounded(answer, candidates):
            return ""
        evidence_logger.record_event(
            event_name="MISRA SEMANTIC RELATION VERIFIER",
            status="SOURCE-GROUNDED VERDICT",
            details={"reference": reference, "verdict": verdict},
        )
        return answer

    def _deterministic_structured_answer(
        self,
        context: str,
        question: str,
        resolved_question: str,
        answer_focus: str,
    ):

        """Return an exact structured answer without invoking an LLM."""

        answer = self._apply_structured_answer_focus(
            context=context,
            question=question,
            resolved_question=resolved_question,
            answer_focus=answer_focus,
            current_answer="",
        )

        if not answer:
            return ""

        answer = self._remove_invalid_yes_no_prefix(
            answer,
            answer_focus,
        )

        return self._apply_output_safety_gate(
            answer,
            question,
        )


    @staticmethod
    def _clean_structured_statement_for_display(statement: str):
        """Remove source-reference noise that is not part of the user-facing requirement.

        The underlying structured source block remains untouched.  This helper
        only shapes the displayed answer so references such as C90/C99 undefined
        behaviour mappings, standards cross-references, or bibliography markers
        do not appear unless the user explicitly asks for applicability/source
        metadata.
        """

        text = re.sub(r"\s+", " ", str(statement or "")).strip()
        if not text:
            return ""

        # Remove trailing C90/C99 technical mapping bundles one at a time.
        # Looping is intentional because many MISRA statements contain both.
        previous = None
        while previous != text:
            previous = text
            text = re.sub(
                r"\s+C(?:90|99)\s*\[[^\]]+\]\s*[,;]?\s*$",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()

        # Remove trailing bibliography / external-standard reference blocks.
        # Keep syntax-bearing brackets such as "between the [ ]" because these
        # do not match the known reference prefixes below.
        previous = None
        while previous != text:
            previous = text
            text = re.sub(
                r"\s*,?\s*\[(?:IEC\b|ISO\b|DO-\d|Koenig\b|"
                r"Undefined\b|Unspecified\b|Implementation\b)[^\]]*\]\s*$",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()

        text = re.sub(r"\bnon-\s+", "non-", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        return text.rstrip(" ;,")


    @staticmethod
    def _first_meaningful_rationale_sentence(raw_text: str):
        """Return a concise rationale without overstating one numbered item.

        Some MISRA Rules contain several numbered rationale points. Returning
        only item ``1`` without qualification makes that first point look like
        the complete rationale. For multi-item rationale blocks, keep the
        answer concise while clearly identifying the displayed sentence as the
        first of several source points.
        """

        text = re.sub(r"\s+", " ", str(raw_text or "")).strip()
        if not text:
            return ""

        numbered_items = list(
            re.finditer(
                r"(?:^|\s)(?P<number>\d+)\.\s+(?=\S)",
                text,
            )
        )
        has_multiple_numbered_points = bool(
            len(numbered_items) >= 2
            and numbered_items[0].group("number") == "1"
        )

        text = re.sub(
            r"^\s*(?:\(?\d+\)?(?:[.)]|\s+))\s*",
            "",
            text,
        ).strip()
        if not text:
            return ""

        first_sentence = ""
        for part in re.split(r"(?<=[.!?])\s+", text):
            candidate = re.sub(
                r"^\s*(?:\(?\d+\)?(?:[.)]|\s+))\s*",
                "",
                str(part or ""),
            ).strip()
            if not candidate:
                continue
            if re.fullmatch(r"\(?\d+\)?[.)]?", candidate):
                continue
            first_sentence = candidate
            break

        if not first_sentence:
            first_sentence = text

        if has_multiple_numbered_points:
            return (
                "The source provides multiple numbered rationale points; "
                f"the first states: {first_sentence}"
            )

        return first_sentence


    @staticmethod
    def _compress_rule_ids_for_display(rule_ids):
        """Compress exact Rule identifiers into compact major-family ranges."""

        parsed = []
        for raw in rule_ids or []:
            value = str(raw or "").strip()
            match = re.fullmatch(r"(\d+)\.(\d+)", value)
            if not match:
                continue
            parsed.append((int(match.group(1)), int(match.group(2))))

        if not parsed:
            return []

        by_major = {}
        for major, minor in sorted(set(parsed)):
            by_major.setdefault(major, []).append(minor)

        output = []
        for major in sorted(by_major):
            minors = sorted(set(by_major[major]))
            ranges = []
            start = previous = minors[0]
            for minor in minors[1:]:
                if minor == previous + 1:
                    previous = minor
                    continue
                ranges.append((start, previous))
                start = previous = minor
            ranges.append((start, previous))

            parts = []
            count = 0
            for first, last in ranges:
                if first == last:
                    parts.append(f"{major}.{first}")
                    count += 1
                else:
                    parts.append(f"{major}.{first}–{major}.{last}")
                    count += last - first + 1

            prefix = "Rule" if count == 1 else "Rules"
            output.append(f"{prefix} " + ", ".join(parts))

        return output


    @staticmethod
    def _deterministic_structured_major_family_answer(results):
        """Explain a bare major Rule family from authoritative child entries.

        A request such as ``Rule 17`` does not identify one MISRA Rule when the
        indexed source contains ``17.1``, ``17.2``, and so on.  Give a compact
        grounded family overview first, then ask for the exact child Rule only
        when the user wants its individual rationale/details.
        """

        for item in results or []:
            if not isinstance(item, dict) or not item.get("_structured_major_family_anchor"):
                continue

            major = str(item.get("_structured_major_family_id", "") or "").strip()
            members = [
                str(value or "").strip()
                for value in item.get("_structured_major_family_members", [])
                if str(value or "").strip()
            ]
            text = str(item.get("text", "") or "").strip()
            if not major or not members or not text:
                continue

            groups = AnswerService._compress_rule_ids_for_display(members)
            family_label = (
                ", ".join(groups)
                if groups
                else ", ".join(f"Rule {m}" for m in members)
            )

            # Split the merged authoritative family block into child Rule
            # segments and preserve only each requirement statement/category.
            starts = list(
                re.finditer(
                    rf"(?im)^\s*Rule\s+({re.escape(major)}\.\d+)\s*$",
                    text,
                )
            )
            overview = []
            for index, match in enumerate(starts):
                rule_id = match.group(1)
                if rule_id not in members:
                    continue
                segment_end = (
                    starts[index + 1].start()
                    if index + 1 < len(starts)
                    else len(text)
                )
                segment = text[match.end():segment_end].strip()
                lines = [
                    line.strip()
                    for line in segment.splitlines()
                    if line.strip()
                ]

                statement_parts = []
                for line in lines:
                    if re.fullmatch(
                        r"(?i)(?:Category|Analysis|Applies to|Rationale|"
                        r"Amplification|Example|Examples|Exception|Exceptions|"
                        r"See also)",
                        line,
                    ):
                        break
                    statement_parts.append(line)
                statement = re.sub(
                    r"\s+",
                    " ",
                    " ".join(statement_parts),
                ).strip()
                statement = AnswerService._clean_structured_statement_for_display(
                    statement
                )

                category_match = re.search(
                    r"(?im)^\s*Category\s*$\s*"
                    r"^\s*(Mandatory|Required|Advisory)\s*$",
                    segment,
                )
                category = category_match.group(1) if category_match else ""

                if statement:
                    label = f"**Rule {rule_id}"
                    if category:
                        label += f" — {category}"
                    label += "**"
                    overview.append(
                        f"- {label}: {statement.rstrip('.')}."
                    )

            intro = (
                f"There is no single **Rule {major}** entry. "
                f"The source contains the Rule {major} family: "
                f"**{family_label}**."
            )
            if overview:
                intro += "\n\n" + "\n".join(overview)

            intro += (
                f"\n\nA request for the rationale of **Rule {major}** is "
                f"ambiguous because these are separate Rule {major}.x entries. "
                f"Specify the exact Rule {major}.x if you want its individual "
                f"rationale or examples."
            )
            return intro

        return ""


    @staticmethod
    def _deterministic_structured_topic_list_answer(results):
        """Render complete structured topic-family results without an LLM."""

        items = []
        for item in results or []:
            if not isinstance(item, dict) or not item.get("_structured_topic_anchor"):
                continue
            metadata = item.get("metadata", {}) or {}
            rule_id = str(metadata.get("rule_id", "") or "").strip()
            text = str(item.get("text", "") or "").strip()
            if not rule_id or not text:
                continue
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            statement_parts = []
            for line in lines[1:]:
                if re.fullmatch(r"(?i)(?:Category|Analysis|Applies to|Rationale|Amplification|Example|Exception|See also)", line):
                    break
                statement_parts.append(line)
            statement = re.sub(r"\s+", " ", " ".join(statement_parts)).strip()
            statement = AnswerService._clean_structured_statement_for_display(statement)
            if statement:
                items.append((rule_id, statement))

        if not items:
            return ""

        compact_ids = any(
            bool(item.get("_structured_topic_compact_ids"))
            for item in (results or [])
            if isinstance(item, dict)
        )
        scope_note = next(
            (
                str(item.get("_structured_topic_scope_note", "") or "").strip()
                for item in (results or [])
                if isinstance(item, dict)
                and str(item.get("_structured_topic_scope_note", "") or "").strip()
            ),
            "",
        )

        def key(value):
            try:
                return tuple(int(part) for part in value[0].split("."))
            except ValueError:
                return (9999,)

        items.sort(key=key)
        family = next(
            (
                str(item.get("_structured_topic_family", "") or "").strip()
                for item in (results or [])
                if isinstance(item, dict) and item.get("_structured_topic_family")
            ),
            "",
        )
        explicit_heading = next(
            (
                str(item.get("_structured_topic_heading", "") or "").strip()
                for item in (results or [])
                if isinstance(item, dict) and item.get("_structured_topic_heading")
            ),
            "",
        )
        display_family = family[:-1] if family.lower().endswith("s") else family
        heading_text = explicit_heading or (f"{display_family.capitalize()} rules" if display_family else "")
        heading = f"### {heading_text}\n\n" if heading_text else ""

        if compact_ids:
            groups = AnswerService._compress_rule_ids_for_display(
                [rule_id for rule_id, _statement in items]
            )
            note = f"{scope_note}\n\n" if scope_note else ""
            body = "\n".join(f"- **{group}**" for group in groups)
            return heading + note + body

        body = "\n".join(
            f"- **Rule {rule_id}** — {statement.rstrip('.')} .".replace(" .", ".")
            for rule_id, statement in items
        )
        note = f"{scope_note}\n\n" if scope_note else ""
        return heading + note + body

    @staticmethod
    def _deterministic_structured_comparison_answer(results):
        """Compare multiple explicit Rule/Directive anchors without neighbor noise."""

        anchors = []
        seen = set()
        for item in results or []:
            if not isinstance(item, dict) or not item.get("_structured_comparison_anchor"):
                continue
            metadata = item.get("metadata", {}) or {}
            reference = str(
                item.get("_structured_comparison_reference")
                or metadata.get("exact_reference")
                or ""
            ).strip()
            if not reference or reference in seen:
                continue
            statement = MisraComplianceMode._source_rule_statement(
                str(item.get("text", "") or "")
            )
            statement = re.sub(r"\s+", " ", statement).strip().rstrip(".")
            statement = AnswerService._clean_structured_statement_for_display(statement)
            if not statement:
                continue
            seen.add(reference)
            anchors.append((reference, statement))

        if len(anchors) < 2:
            return ""

        heading = "### Comparison\n\n"
        bullets = "\n".join(
            f"- **{reference}** — {statement}."
            for reference, statement in anchors
        )
        if len(anchors) == 2:
            (left_ref, left_statement), (right_ref, right_statement) = anchors
            relation = (
                f"\n\n**Difference:** {left_ref} governs this requirement: "
                f"{left_statement}. In contrast, {right_ref} governs this requirement: "
                f"{right_statement}."
            )
        else:
            relation = "\n\n**Difference:** " + " ".join(
                f"{reference} governs this requirement: {statement}."
                for reference, statement in anchors
            )
        return heading + bullets + relation

    @staticmethod
    def _deterministic_structured_multi_reference_answer(results, question: str = ""):
        """Explain multiple explicit Rule/Directive anchors without metadata dumping."""

        clean_question = re.sub(r"\s+", " ", str(question or "").lower()).strip()
        include_category = bool(
            re.search(
                r"\b(?:category|classification|classified|mandatory|required|advisory)\b",
                clean_question,
            )
        )

        anchors = []
        seen = set()
        for item in results or []:
            if not isinstance(item, dict) or not item.get("_structured_comparison_anchor"):
                continue
            metadata = item.get("metadata", {}) or {}
            reference = str(
                item.get("_structured_comparison_reference")
                or metadata.get("exact_reference")
                or ""
            ).strip()
            if not reference or reference in seen:
                continue
            text = str(item.get("text", "") or "")
            statement = re.sub(
                r"\s+", " ", MisraComplianceMode._source_rule_statement(text)
            ).strip().rstrip(".")
            statement = AnswerService._clean_structured_statement_for_display(statement)
            if not statement:
                continue

            category = ""
            category_match = re.search(
                r"(?im)^\s*Category\s*$\s*^\s*(Mandatory|Required|Advisory)\s*$",
                text,
            )
            if category_match:
                category = category_match.group(1).strip()

            rationale = ""
            rationale_match = re.search(
                r"(?ims)^\s*Rationale\s*$\s*(.+?)(?=^\s*(?:Amplification|Example|Examples|Exception|Exceptions|Note|Notes|See\s+also|Category|Analysis|Applies\s+to)\s*$|\Z)",
                text,
            )
            if rationale_match:
                rationale = AnswerService._first_meaningful_rationale_sentence(
                    rationale_match.group(1)
                )

            seen.add(reference)
            anchors.append((reference, category, statement, rationale))

        if len(anchors) < 2:
            return ""

        pieces = []
        for reference, category, statement, rationale in anchors:
            heading = f"**{reference}"
            if include_category and category:
                heading += f" — {category}"
            heading += "**"
            body = f"{heading}\n- {statement.rstrip('.')}."
            if rationale:
                body += f"\n- **Rationale:** {rationale}"
            pieces.append(body)
        return "\n\n".join(pieces)

    @staticmethod
    def _rationale_cross_reference_ids(text: str):
        """Return exact Section identifiers explicitly cited by a rationale."""

        if not text:
            return []

        ids = []
        seen = set()
        for match in re.finditer(
            r"\bSection\s+([A-Za-z0-9IVXLCDM]+(?:\.\d+)*)\b",
            str(text),
            re.IGNORECASE,
        ):
            identifier = match.group(1).strip().casefold()
            if identifier and identifier not in seen:
                seen.add(identifier)
                ids.append(identifier)
        return ids


    def _rationale_cross_reference_answer(
        self,
        context: str,
        rationale: str,
    ):
        """Expand a source-deferred rationale from its exact cited Section.

        Some Rule rationale blocks intentionally contain only a pointer such as
        "see Section 8.10.3".  When that cited Section has already been added to
        accepted context, answer the user's why/rationale request from the
        Section itself instead of echoing only the pointer sentence.
        """

        section_ids = self._rationale_cross_reference_ids(rationale)
        if not section_ids:
            return ""

        for section_id in section_ids:
            reference = StructuredReference(kind="section", identifier=section_id)
            title, intro = self._section_heading_and_intro_from_context(
                context,
                reference,
            )
            points = self._section_body_key_points_from_context(
                context,
                reference,
                max_points=2,
            )
            if not intro and not points:
                continue

            heading = f"**Rationale — Section {section_id}"
            if title:
                heading += f": {title}"
            heading += "**"

            pieces = [heading]
            if intro:
                intro_lead = re.split(
                    r"\b(?:These|They)\s+include\s*:",
                    intro,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0].strip()
                intro_lead = self._limit_grounded_explanation_text(
                    intro_lead,
                    max_sentences=1,
                    max_chars=420,
                )
                if intro_lead:
                    pieces.append(intro_lead.rstrip(".") + ".")
            if points:
                pieces.append(
                    "\n".join(f"- {point.rstrip('.') }." for point in points)
                )
            return "\n\n".join(piece for piece in pieces if piece).strip()

        return ""


    @staticmethod
    def _is_rule_causal_intent(text: str):
        """Recognize source-rationale intent across natural paraphrases.

        Keep this vocabulary centralized so answer-focus routing and exact
        rationale cross-reference augmentation cannot drift apart.

        This detector is only acted on for an already-resolved exact
        Rule/Directive reference; it does not broaden retrieval scope.
        """

        clean = re.sub(r"\s+", " ", str(text or "").casefold()).strip()
        if not clean:
            return False

        patterns = (
            r"\bwhy\b",
            r"\brationales?\b",
            r"\breasons?\b",
            r"\bpurposes?\b",
            r"\bjustifications?\b",
            r"\bbasis\b",
            r"\bbases\b",
            r"\bgrounds?\b",
            r"\bmotivat(?:e|ed|es|ing|ion|ions)\b",
            r"\bgoals?\b",
            r"\baims?\b",
            r"\bobjectives?\b",
            r"\bintent\b",
            r"\btrying\s+to\s+achieve\b",
            r"\bbakit\b",
            r"\bdahilan\b",
            r"\blayunin\b",
        )
        return any(re.search(pattern, clean, flags=re.IGNORECASE) for pattern in patterns)


    def _augment_reason_cross_reference_context(
        self,
        context: str,
        results,
        question: str,
    ):
        """Append exact Sections explicitly cited by a retrieved rationale.

        This is a source-following operation, not semantic expansion: the
        Section identifier must literally appear in the accepted Rule/Directive
        rationale.  It therefore improves completeness without weakening the
        exact-reference grounding boundary.
        """

        clean_question = re.sub(r"\s+", " ", str(question or "").casefold()).strip()
        if not self._is_rule_causal_intent(clean_question):
            return context, list(results or [])

        base_results = [item for item in (results or []) if isinstance(item, dict)]
        if not base_results:
            return context, base_results

        referenced_ids = []
        for item in base_results:
            text = str(item.get("text", "") or "")
            rationale_match = re.search(
                r"(?ims)^\s*Rationale\s*$\s*(.+?)"
                r"(?=^\s*(?:Amplification|Example|Examples|Exception|Exceptions|"
                r"Note|Notes|See\s+also|Category|Analysis|Applies\s+to)\s*$|\Z)",
                text,
            )
            if not rationale_match:
                continue
            referenced_ids.extend(
                self._rationale_cross_reference_ids(rationale_match.group(1))
            )

        referenced_ids = list(dict.fromkeys(referenced_ids))
        if not referenced_ids:
            return context, base_results

        try:
            retriever = self.query_service._get_retriever()
        except Exception:
            return context, base_results

        existing_keys = {
            (
                str((item.get("metadata", {}) or {}).get("file_path", "") or ""),
                str((item.get("metadata", {}) or {}).get("section_id", "") or "").casefold(),
            )
            for item in base_results
        }
        appended = []

        parent_file_paths = {
            str((item.get("metadata", {}) or {}).get("file_path", "") or "")
            for item in base_results
            if str((item.get("metadata", {}) or {}).get("file_path", "") or "")
        }

        for section_id in referenced_ids[:3]:
            reference = StructuredReference(kind="section", identifier=section_id)
            try:
                exact = retriever._retrieve_exact_structured_reference(reference)
            except Exception:
                exact = []

            for item in exact:
                if not isinstance(item, dict):
                    continue
                metadata = dict(item.get("metadata", {}) or {})
                file_path = str(metadata.get("file_path", "") or "")
                if parent_file_paths and file_path and file_path not in parent_file_paths:
                    continue
                key = (file_path, section_id.casefold())
                if key in existing_keys:
                    continue
                copied = dict(item)
                copied["metadata"] = metadata
                copied["_rationale_cross_reference_anchor"] = True
                copied["_rationale_cross_reference_id"] = section_id
                appended.append(copied)
                existing_keys.add(key)

        if not appended:
            return context, base_results

        combined = base_results + appended
        combined_context = ""
        for index, item in enumerate(combined, start=1):
            combined_context += (
                f"\n\n===== DOCUMENT {index} =====\n"
                f"{str(item.get('text', '') or '')}"
            )

        evidence_logger.record_event(
            event_name="RATIONALE CROSS-REFERENCE FOLLOW",
            status="EXACT SECTION APPENDED",
            details={
                "section_ids": referenced_ids[:3],
                "appended_sections": [
                    str((item.get("metadata", {}) or {}).get("section_id", "") or "")
                    for item in appended
                ],
            },
        )
        evidence_logger.record_context(
            context=combined_context,
            chunk_count=len(combined),
        )
        return combined_context, combined


    def _deterministic_named_section_answer(self, results):
        """Render a near-exact Section-title lookup without LLM generation."""

        for item in results or []:
            if not isinstance(item, dict) or not item.get("_structured_section_title_anchor"):
                continue
            metadata = item.get("metadata", {}) or {}
            section_id = str(metadata.get("section_id", "") or "").strip()
            text = str(item.get("text", "") or "").strip()
            if not section_id or not text:
                continue

            reference = StructuredReference(kind="section", identifier=section_id)
            title, intro = self._section_heading_and_intro_from_context(
                text,
                reference,
            )

            # Compiler/toolchain switch/option questions are best answered as
            # configuration guidance from the matched Section, not as an
            # unrelated MISRA Rule inventory.  Pair consecutive source
            # sentences so the original paragraph relationships remain intact.
            if item.get("_structured_toolchain_configuration_anchor"):
                clean_body = re.sub(
                    rf"(?im)^\s*(?:Section\s+)?{re.escape(section_id)}\b[^\n]*\n?",
                    "",
                    text,
                    count=1,
                ).strip()

                # A recovered continuation can carry a repeated ancestor heading.
                clean_body = re.sub(
                    r"(?im)^\s*Section\s+\d+(?:\.\d+)*\s*:[^\n]*\n?",
                    "",
                    clean_body,
                ).strip()

                sentences = self._presentation_sentences(clean_body)
                groups = []
                for index in range(0, min(len(sentences), 6), 2):
                    group = " ".join(sentences[index:index + 2]).strip()
                    if group:
                        groups.append(group.rstrip(".") + ".")

                heading = f"**Section {section_id}"
                if title:
                    heading += f" — {title}"
                heading += "**"

                if groups:
                    bullets = "\n".join(f"- {group}" for group in groups)
                    return (
                        f"{heading}\n\n"
                        f"Key compiler configuration guidance from the source:\n\n"
                        f"{bullets}"
                    )

            if not intro:
                intro = self._limit_grounded_explanation_text(
                    re.sub(
                        rf"(?im)^\s*(?:Section\s+)?{re.escape(section_id)}\b[^\n]*\n?",
                        "",
                        text,
                        count=1,
                    ),
                    max_sentences=2,
                    max_chars=650,
                )
            if not intro:
                continue

            heading = f"**Section {section_id}"
            if title:
                heading += f" — {title}"
            heading += "**"
            return f"{heading}\n\n{intro.rstrip('.')}."

        return ""


    def _structured_summary_from_context(self, context: str, reference, display_name: str):
        """Return only the concise requirement and rationale requested by a summary."""

        statement = self._structured_statement_from_context(context, reference)
        statement = self._clean_structured_statement_for_display(statement)
        if not statement:
            return ""

        rationale = self._structured_rationale_from_context(context)
        rationale = self._first_meaningful_rationale_sentence(rationale)

        pieces = [
            f"**{display_name}**",
            statement.rstrip(".") + ".",
        ]
        if rationale:
            pieces.append(f"**Rationale:** {rationale}")
        return "\n\n".join(pieces)

    def _build_chat_history(self):

        """
        Build conversation history from the current chat session.

        Important:
        - Conversation history is used only for follow-up reference.
        - Do not include full assistant answers because they can contaminate
        the next response language or facts.
        """

        messages = (
            ChatManager.get_current_messages()
        )

        # Exclude the current user message from history.
        # The current question is already inserted separately
        # as USER QUESTION in the final prompt.
        if (
            messages
            and messages[-1].get("role") == "user"
        ):

            messages = messages[:-1]

        # Keep only the last 6 messages
        messages = messages[-6:]

        history = []

        for message in messages:

            role = message.get(
                "role",
                ""
            )

            content = message.get(
                "content",
                ""
            )

            # Keep user questions.
            if role == "user":

                history.append(
                    f"User: {content}"
                )

            # Do not include assistant answers.
            # They are not source of truth and can contaminate the next answer.
            elif role == "assistant":

                continue

        return "\n".join(history)

    @staticmethod
    def _compiler_switch_ambiguity_clarification(question: str) -> str:
        """Return a clarification for generic compiler switch/flag wording.

        A compiler command-line switch/flag and the document's broader compiler
        configuration guidance are related engineering concepts, but they are
        not interchangeable.  Only broad, unnamed switch/flag questions are
        intercepted here.  Explicit configuration/options questions and named
        command-line switches remain on normal source-grounded retrieval paths.
        """

        raw = str(question or "").strip()
        clean = re.sub(r"\s+", " ", raw.casefold()).strip()
        if not clean:
            return ""

        if re.search(
            r"\bswitch\s+(?:statement|statements|case|cases|clause|clauses)\b",
            clean,
        ):
            return ""

        if re.search(
            r"#\s*(?:if|elif|ifdef|ifndef)\b|\bpreprocess(?:or|ing)\b",
            raw,
            flags=re.IGNORECASE,
        ):
            return ""

        if not re.search(r"\b(?:compiler|toolchain|build(?:\s+tool)?)\b", clean):
            return ""

        if not re.search(r"\b(?:switch|switches|flag|flags)\b", clean):
            return ""

        # Only explicit configuration wording resolves the ambiguity itself.
        # Words such as "options" or "settings" do not make a simultaneous
        # switch/flag request unambiguous because engineers often use those
        # words when referring to command-line flags too.
        if re.search(
            r"\b(?:compiler\s+configuration|configuration\s+of\s+the\s+compiler|"
            r"configure|configured|setup)\b",
            clean,
        ):
            return ""

        # A named command-line token should be looked up normally rather than
        # replaced with a generic clarification.
        if re.search(
            r"(?<![A-Za-z0-9_])(?:--?[A-Za-z0-9][A-Za-z0-9_-]*|/[A-Za-z][A-Za-z0-9_-]*)\b",
            raw,
        ):
            return ""

        return (
            "'Compiler switch/flag' is ambiguous here. Do you mean a specific "
            "compiler command-line switch/flag, or the document's general "
            "compiler configuration guidance? Please clarify which one you want."
        )

    @staticmethod
    def _is_grounded_followup_candidate(question: str, state) -> bool:
        """Recognize conservative references to the immediately grounded turn.

        The anchor is intentionally narrow: a prior accepted state must exist and
        the new question must contain a deictic/reference phrase such as this/that/
        it/yan/ito/same document, or an explicit request to justify/fix the prior
        answer.  New standalone topics are left on the normal retrieval path.
        """

        if not isinstance(state, dict) or not state.get("context"):
            return False

        clean = re.sub(r"\s+", " ", str(question or "").strip().casefold())
        if not clean:
            return False

        # In phrases such as "Is it OK to use trigraphs?", "it" is an
        # expletive grammatical subject, not a reference to the prior turn.
        # Treat a concrete permission/compliance action as standalone unless
        # the action itself is only deictic (for example "do that").
        permission = re.match(
            r"^is\s+it\s+(?:ok(?:ay)?|allowed|permitted|acceptable|safe|"
            r"valid|compliant|required|mandatory|necessary|advisable|recommended)"
            r"\s+to\s+(.+)$",
            clean,
            flags=re.IGNORECASE,
        )
        if permission:
            action = re.sub(r"[?!.]+$", "", permission.group(1)).strip()
            if not re.fullmatch(
                r"(?:(?:do|use|apply|change|remove|keep|allow)\s+)?"
                r"(?:it|this|that|these|those|same)(?:\s+(?:thing|one|ones))?",
                action,
                flags=re.IGNORECASE,
            ):
                return False

        # Freshness/current-state questions are standalone requests, not
        # continuations of the previous technical topic. This prevents stale
        # switch context from leaking into questions such as "What is the
        # newest MISRA rule released this year?".
        if re.search(
            r"\b(?:latest|newest|current|today|this\s+year|released\s+this\s+year|most\s+recent)\b",
            clean,
            re.IGNORECASE,
        ):
            return False

        # An explicit current-turn structured identifier is self-contained.
        # Never reinterpret ``Rule 99.99`` / ``Directive 4.12`` as a deictic
        # continuation merely because words such as ``requirement`` also occur.
        # This keeps exact identifiers authoritative over stale grounded state.
        if extract_structured_reference(question) is not None:
            return False

        # A self-contained technical/MISRA concept is a new request even when
        # it starts with a follow-up-looking word such as Why/Bakit.  Only an
        # explicit anaphor (this/that/it/ito/yan/...) may bind such a turn to the
        # prior grounded state.  This prevents stale Rule/topic reuse across
        # consecutive technical questions.
        explicit_anaphor = bool(re.search(
            r"\b(?:this|that|it|its|same|ito|iyan|iyon|yan|yun|nito|niyan|niyon|"
            r"siya|niya|ganito|ganyan)\b",
            clean,
            re.IGNORECASE,
        ))
        if state.get("misra") is True and not explicit_anaphor:
            if MisraComplianceMode.semantic_cues(question):
                return False
            if re.search(
                r"(?:&&|\|\|)|\b(?:side[- ]?effect|recursion|recursive|pointer|"
                r"parameter|switch|goto|label|string|stdarg|standard\s+library|"
                r"macro|operator|declaration)\b",
                str(question or ""),
                re.IGNORECASE,
            ):
                return False

        # A new MISRA scenario may contain deictic words such as ``iyon`` or
        # ``that`` while still being fully self-contained (for example a
        # text-only question describing a function call on the right-hand side
        # of &&). If the current turn independently produces a source-language
        # MISRA cue, do not borrow the previous grounded scenario. This keeps
        # follow-up memory narrow without hardcoding any Rule/Directive number.
        if state.get("misra") is True:
            if (
                MisraComplianceMode.looks_like_c_cpp(question)
                or re.search(
                    r"#\s*(?:include|define|undef|if|elif|ifdef|ifndef|endif|pragma)\b",
                    str(question or ""),
                    re.IGNORECASE,
                )
            ):
                return False
            if MisraComplianceMode.semantic_cues(question):
                return False

        if state.get("misra") is True and re.match(
            r"^(?:so\b|what\s+about\b|how\s+about\b|what\s+if\b|and\s+if\b|"
            r"does\s+that\b|is\s+that\b|can\s+that\b)",
            clean,
            re.IGNORECASE,
        ):
            return True

        reference_signal = re.search(
            r"\b(?:this|that|it|its|same\s+(?:document|source|code|snippet)|"
            r"ito|iyan|iyon|yan|yun|nito|niyan|niyon|dito|diyan|doon|dun|"
            r"siya|niya|ganito|ganyan|yung|based\s+(?:on|sa)\s+(?:the\s+)?document|same\s+doc)\b",
            clean,
            re.IGNORECASE,
        )
        followup_signal = re.search(
            r"^(?:which\s+part|what\s+other|ano\s+pa|may\s+iba|aling\s+part|"
            r"kung\s+aayusin|does\s+that|is\s+that|can\s+you\s+explain\s+that|"
            r"explain\b.*\b(?:this|that|it|ito|iyan|iyon|yan|yun)\b|"
            r"may\s+(?:example|halimbawa)\s+(?:nito|niyan|niyon|dito|yan|yun|ito)\b|"
            r"(?:okay\s*[,;:]?\s*)?sino\s+(?:naman\s+)?(?:ang\s+)?(?:nag[- ]?aapprove|nag[- ]?approve|umaapprove|mag[- ]?approve)"
            r"\s*(?:nito|niyan|niyon|dito|diyan|doon|dun|yan|yun|ito)\s*[?!.]*$|"
            r"who\s+(?:else\s+)?approves?\s+(?:it|this|that|them|these|those)\s*[?!.]*$)",
            clean,
            re.IGNORECASE,
        )
        if reference_signal or followup_signal:
            return True

        # A compact detail request is a follow-up only when it contains an
        # explicit deictic reference.  Bare questions such as "Why should ...",
        # "What rule covers ...", or "May rule ba sa ..." are standalone new
        # retrieval requests even when they immediately follow another MISRA
        # turn.  This prevents stale grounded-state contamination.
        if (
            reference_signal
            and len(clean.split()) <= 9
            and re.search(
                r"\b(?:example|halimbawa|requirements?|rule|directive|rationale|"
                r"reason|basis|source|document)\b",
                clean,
                re.IGNORECASE,
            )
        ):
            return True

        return False

    @staticmethod
    def _question_has_visible_code(question: str) -> bool:
        return MisraComplianceMode.looks_like_c_cpp(question)

    def _store_grounded_followup_state(
        self,
        *,
        question: str,
        context: str,
        results,
        misra_compliance_mode: bool,
        answer_focus: str,
        previous_state=None,
        grounded_followup: bool = False,
    ) -> None:
        """Persist only accepted same-chat evidence needed for the next follow-up."""

        if not context or not results:
            return


        anchor_question = str(question or "").strip()
        if (
            grounded_followup
            and isinstance(previous_state, dict)
            and previous_state.get("anchor_question")
            and not self._question_has_visible_code(anchor_question)
            and not (
                bool(misra_compliance_mode)
                and bool(MisraComplianceMode.semantic_cues(anchor_question))
            )
        ):
            anchor_question = str(previous_state.get("anchor_question") or "").strip()

        # Keep a compact serializable copy. Retrieval result dictionaries are
        # already plain data; copying prevents later mutation from changing the
        # stored evidence unexpectedly.
        result_list = [item for item in list(results or []) if isinstance(item, dict)]
        preserve_complete_structured_set = bool(result_list) and all(
            item.get("_structured_topic_anchor") or item.get("_structured_comparison_anchor")
            for item in result_list
        )
        state_limit = len(result_list) if preserve_complete_structured_set else 6

        compact_results = []
        references = []
        for item in result_list[:state_limit]:
            copied = dict(item)
            if isinstance(copied.get("metadata"), dict):
                copied["metadata"] = dict(copied["metadata"])
            compact_results.append(copied)
            metadata = copied.get("metadata", {}) or {}
            rule_id = str(metadata.get("rule_id") or "").strip()
            directive_id = str(metadata.get("directive_id") or "").strip()
            if rule_id:
                references.append(f"Rule {rule_id}")
            elif directive_id:
                references.append(f"Directive {directive_id}")

        ChatManager.set_grounded_state({
            "anchor_question": anchor_question,
            "last_question": str(question or "").strip(),
            "context": str(context or ""),
            "results": compact_results,
            "references": list(dict.fromkeys(references)),
            "misra": bool(misra_compliance_mode),
            "answer_focus": str(answer_focus or ""),
        })

    def _rewrite_question(
        self,
        question,
        history
    ):

        """
        Rewrite follow-up questions into
        standalone questions before retrieval.

        Currently kept for future fallback use.
        """

        if not history.strip():

            return question

        prompt = REWRITE_QUERY_PROMPT.format(
            history=history,
            question=question
        )

        try:

            auxiliary_client = self._auxiliary_llm_for_question(
                question
            )
            rewritten = self._generate_with_latency(
                auxiliary_client,
                prompt,
                purpose="followup_query_rewrite",
            )

            rewritten = rewritten.strip()

            if not rewritten:

                return question

            return rewritten

        except Exception:

            return question

    def _is_compound_question(
        self,
        question: str
    ):

        """
        Detect questions that contain two or more independently requested parts.

        Besides classic ``what ... and who ...`` wording, users often combine
        a direct lookup with an imperative follow-up in the same turn, for
        example ``What is Rule 13.5 and explain it as well`` or
        ``Ano ang X? Ipaliwanag din ito``. Those are multi-intent requests even
        though the second clause has no interrogative word.
        """

        if not question:

            return False

        clean = re.sub(
            r"\s+",
            " ",
            question.lower().strip()
        )

        interrogatives = (
            r"(?:who|what|when|where|why|how|which|"
            r"sino|ano|anong|kailan|saan|bakit|paano|alin)"
        )
        request_starters = (
            interrogatives
            + r"|(?:explain|describe|summari[sz]e|define|list|enumerate|"
            r"give|show|provide|tell|clarify|elaborate|"
            r"ipaliwanag|ilarawan|ibuod|ilista|ibigay)"
        )

        # A conjunction introduces another explicit question or instruction.
        if re.search(
            rf"\b(?:and|at)\b\s+(?:also\s+|din\s+|rin\s+)?"
            rf"(?:{request_starters})\b",
            clean,
            flags=re.IGNORECASE,
        ):
            return True

        # A sentence/semicolon boundary introduces another request.
        if re.search(
            rf"[?;]\s*(?:please\s+|paki\s*)?(?:{request_starters})\b",
            question,
            flags=re.IGNORECASE,
        ):
            return True

        # Multiple question marks/semicolons remain an unambiguous signal.
        if len(re.findall(r"[?;]", question)) >= 2:
            return True

        return False

    def _looks_like_plural_list_question(
        self,
        question: str
    ):

        """
        Detect generic plural-item questions such as:
        - What hardships...
        - What requirements...
        - Which controls...

        Avoid auxiliary words such as "is", "was", and "does".
        """

        if not question:

            return False

        clean = re.sub(
            r"\s+",
            " ",
            question.lower().strip()
        )

        match = re.match(
            r"^(?:what|which)\s+([a-z][a-z0-9_-]*)\b",
            clean
        )

        if not match:

            return False

        candidate = match.group(1)

        excluded = {
            "is",
            "was",
            "does",
            "has",
            "can",
            "could",
            "should",
            "would",
            "will",
        }

        return (
            candidate not in excluded
            and candidate.endswith("s")
            and len(candidate) > 3
        )

    def _detect_answer_focus(
        self,
        question: str,
        resolved_question: str = ""
    ):

        """
        Detect the requested answer type from the original question
        while preserving the resolved follow-up target.

        This is a mixed strategy:
        - generic intent categories for any document domain
        - dynamic relation targeting for compatibility with short
          questions and follow-up tests

        No document topic, person, policy, or standard is hardcoded.
        """

        if not question:

            return (
                "GENERAL: Return only the directly requested information."
            )

        clean = re.sub(
            r"\s+",
            " ",
            question.lower().strip()
        )

        resolved_clean = re.sub(
            r"\s+",
            " ",
            (resolved_question or "").strip()
        )

        # Preserve the exact resolved relation or target when available.
        target = (
            resolved_clean
            if resolved_clean
            and resolved_clean.lower() != clean
            else question.strip()
        )

        target_instruction = (
            f' The requested relation or target is: "{target}". '
            "Return only information explicitly connected to that exact "
            "relation or target in COMPANY KNOWLEDGE."
        )

        structured_references = extract_structured_references(
            resolved_question or question
        )
        structured_reference = structured_references[0] if structured_references else None

        explicit_rule_references = [
            ref for ref in structured_references
            if ref.kind in {"rule", "directive"}
        ]

        # Broad MISRA inventory/catalog questions such as "Which MISRA rules
        # apply to pointers?" must remain list intents. Without this guard the
        # generic "which ..." classifier can misroute a complete structured
        # family into ENTITY/CHOICE generation and then discard valid evidence.
        if MisraComplianceMode.informational_catalog_query(question):
            return (
                "LIST: Return every relevant explicitly stated Rule or Directive "
                "from the complete structured inventory. Use one Markdown bullet "
                "per item and preserve each identifier."
                + target_instruction
            )

        if (
            len(explicit_rule_references) >= 2
            and re.search(
                r"\b(?:compare|comparison|contrast|difference|different|versus|vs\.?)\b",
                clean,
                flags=re.IGNORECASE,
            )
        ):
            return (
                "STRUCTURED COMPARISON: Compare every explicitly requested Rule "
                "or Directive using only its own authoritative statement. Do not "
                "substitute rationale, examples, or neighboring identifiers."
                + target_instruction
            )

        if len(explicit_rule_references) >= 2 and re.search(
            r"\b(?:explain|describe|discuss|summari[sz]e|summary|tell\s+me\s+about|overview)\b",
            clean,
            flags=re.IGNORECASE,
        ):
            return (
                "STRUCTURED MULTI EXPLANATION: Explain each explicitly requested "
                "Rule or Directive independently from its own authoritative source "
                "block. Preserve the requested identifiers and do not turn an "
                "ordinary multi-item explanation into a comparison."
                + target_instruction
            )

        if structured_reference:

            if (
                structured_reference.kind in {"rule", "directive"}
                and re.search(r"\b(?:summari[sz]e|summary|overview)\b", clean, re.IGNORECASE)
            ):
                return (
                    "STRUCTURED SUMMARY: Summarize the exact requested Rule or "
                    "Directive from its own authoritative statement and rationale. "
                    "Do not include source examples unless the user explicitly asks "
                    "for examples."
                    + target_instruction
                )

            if (
                structured_reference.kind in {"rule", "directive"}
                and re.search(
                    r"\b(?:explain|describe|mean|means|meaning|ipaliwanag|ilarawan)\b",
                    clean,
                    flags=re.IGNORECASE,
                )
                and re.search(r"\b(?:example|examples|sample|illustrat(?:e|ion))\b", clean, re.IGNORECASE)
            ):
                return (
                    "STRUCTURED EXPLANATION WITH EXAMPLE: Explain the exact "
                    "requested Rule or Directive using only its authoritative "
                    "statement and supporting source sections, then include the "
                    "source Example block when one exists. If the source contains "
                    "no explicit Example section, say so instead of inventing one."
                    + target_instruction
                )

            if (
                structured_reference.kind in {"rule", "directive"}
                and re.search(
                    r"\b(?:in\s+(?:simple|plain)\s+(?:terms|english)|simply)\b",
                    clean,
                    flags=re.IGNORECASE,
                )
            ):
                return (
                    "STRUCTURED SIMPLE EXPLANATION: State the exact Rule or "
                    "Directive requirement, then give one concise source-grounded "
                    "plain-language sentence from its own amplification or rationale. "
                    "Do not dump the full source block, examples, or cross-references."
                    + target_instruction
                )

            if (
                structured_reference.kind in {"rule", "directive"}
                and re.search(
                    r"\b(?:explain|describe|discuss|tell\s+me\s+about|overview)\b",
                    clean,
                    flags=re.IGNORECASE,
                )
                and re.search(
                    r"\b(?:category|classification|classified|mandatory|required|advisory)\b",
                    f"{clean} {resolved_clean.lower()}",
                    flags=re.IGNORECASE,
                )
            ):
                return (
                    "STRUCTURED EXPLANATION WITH CATEGORY: Briefly discuss the exact "
                    "Rule or Directive, state its exact Category, and include a concise "
                    "source-grounded rationale when available. Do not dump raw tables, "
                    "all examples, or unrelated cross-references."
                    + target_instruction
                )

            # Exact Rule/Directive causal intent has priority over the
            # generic "explain" route.  Users naturally ask for the same
            # source Rationale using wording such as why, reason, purpose,
            # justification, rationale, dahilan, or bakit.  Treat these as one
            # semantic family so a question like "Explain the purpose of Rule
            # X" does not degrade into a generic Rule explanation.
            if (
                structured_reference.kind in {"rule", "directive"}
                and self._is_rule_causal_intent(
                    f"{clean} {resolved_clean.lower()}"
                )
                and not re.search(
                    r"\b(?:category|classification|classified|mandatory|required|"
                    r"advisory|analysis|amplification|examples?|exceptions?|"
                    r"apply\s+to|applies\s+to|applicability|see\s+also)\b",
                    f"{clean} {resolved_clean.lower()}",
                    flags=re.IGNORECASE,
                )
            ):
                return (
                    "REASON: Return the exact requested Rule or Directive rationale. "
                    "If that source rationale explicitly defers its explanation to "
                    "an exact cited Section, follow that Section and return the "
                    "grounded reason itself rather than only the cross-reference."
                    + target_instruction
                )

            # Mixed same-turn requests such as
            # ``What is Rule 13.5 and explain it as well`` must not be reduced
            # to the direct statement path merely because the question starts
            # with "what". Explanation intent has priority wherever it appears
            # in the user turn. Rule/Directive explanations remain
            # deterministic; Section explanations keep the complex-model path.
            if re.search(
                r"\b(?:explain|describe|mean|means|meaning|ipaliwanag|ilarawan)\b"
                r"|\b(?:in\s+(?:simple|plain)\s+(?:terms|english)|simply)\b",
                clean,
                flags=re.IGNORECASE,
            ):

                return (
                    "STRUCTURED EXPLANATION: Explain the exact requested "
                    "Rule, Directive, or Section (including Article, Chapter, or Part aliases) using the important supported "
                    "content tied to that same identifier. For a Rule or "
                    "Directive, state the exact requirement first, then cover "
                    "relevant amplification or scope and the rationale when "
                    "available. Keep the explanation complete enough to "
                    "understand the requested item without adding unsupported "
                    "information, unrelated neighboring identifiers, or "
                    "cross-reference content as a substitute for the target."
                    + target_instruction
                )

            structured_named_label = self._structured_detail_label(
                clean,
                resolved_clean,
            )
            if structured_reference.is_section_like and structured_named_label:
                return (
                    "STRUCTURED DETAIL: Return only the explicitly labeled "
                    f"{structured_named_label} field from the exact requested "
                    "structured section. Do not infer or relabel nearby prose."
                    + target_instruction
                )

            if (
                structured_reference.is_section_like
                and re.search(
                    r"\b(?:topics?|subtopics?|headings?)\b",
                    clean,
                    flags=re.IGNORECASE,
                )
            ):
                return (
                    "STRUCTURED SECTION TOPICS: Return only the concise child-topic "
                    "headings of the exact requested structured section. Do not dump references, "
                    "glossary entries, or repeated body terms."
                    + target_instruction
                )

            if (
                structured_reference.is_section_like
                and re.search(
                    r"\b(?:overview|summari[sz]e|summary)\b",
                    clean,
                    flags=re.IGNORECASE,
                )
            ):
                return (
                    "STRUCTURED SECTION OVERVIEW: Return the exact requested "
                    "structured heading and its opening supported description. "
                    "Keep it concise and do not add neighboring-section facts."
                    + target_instruction
                )

            if (
                structured_reference.is_section_like
                and (
                    re.search(r"\b(?:about|regarding|concerning)\b", clean)
                    or re.search(r"\btungkol\s+sa\b", clean)
                    or re.search(r"^what\s+does\s+(?:section|article|chapter|part)\b.+\bsay\s+about\b", clean)
                    or re.search(r"^ano\s+ang\s+sinasabi\b.+\b(?:section|article|chapter|part)\b", clean)
                    or re.search(r"\b(?:distinguish|difference)\b.*\brules?\b.*\bdirectives?\b", clean)
                    or re.search(r"\brules?\b.*\b(?:versus|vs\.?|and)\b.*\bdirectives?\b", clean)
                )
            ):
                return (
                    "STRUCTURED SECTION DETAIL: Return only the part of the exact "
                    "requested structured section that directly addresses the user's named "
                    "subtopic. Do not substitute the Section introduction or a "
                    "neighboring subsection when a closer supported passage exists."
                    + target_instruction
                )

            if (
                structured_reference.is_section_like
                and re.search(
                    r"\b(?:decidab(?:le|ility)|undecidable|categor(?:y|ies)|automatically\s+generated|generated\s+code|scope|applicability)\b",
                    f"{clean} {resolved_clean.lower()}",
                    flags=re.IGNORECASE,
                )
            ):
                return (
                    "STRUCTURED SECTION DETAIL: Return only the part of the exact "
                    "requested structured section that directly addresses the user's named "
                    "subtopic. Do not substitute the Section introduction or a "
                    "neighboring subsection when a closer supported passage exists."
                    + target_instruction
                )

            if (
                structured_reference.is_section_like
                and re.search(
                    r"^(?:what\s+is|what\s+does)\s+(?:section|article|chapter|part)\b",
                    clean
                )
                and not re.search(
                    r"\b(?:compare|summari[sz]e|why|how)\b",
                    clean
                )
            ):

                return (
                    "STRUCTURED SECTION OVERVIEW: Return the exact requested "
                    "structured heading and its opening supported description. "
                    "Keep it concise and do not add neighboring-section facts."
                    + target_instruction
                )

            if structured_reference.kind in {"rule", "directive"}:

                structured_intent_text = f"{clean} {resolved_clean.lower()}"

                if re.search(
                    r"\b(?:category|classification|classified|mandatory|required|advisory)\b",
                    structured_intent_text,
                ):
                    return (
                        "STRUCTURED DETAIL: Return the exact Category value "
                        "from the requested Rule or Directive."
                        + target_instruction
                    )

                if re.search(
                    r"\b(?:apply\s+to|applies\s+to|applicable\s+to|applicability|c\s+versions?|c\s+standards?)\b",
                    structured_intent_text
                ):
                    return (
                        "STRUCTURED DETAIL: Return the exact Applies to value "
                        "from the requested Rule or Directive."
                        + target_instruction
                    )

                if re.search(r"\banalysis\b", structured_intent_text):
                    return (
                        "STRUCTURED DETAIL: Return the exact Analysis value "
                        "from the requested Rule or Directive."
                        + target_instruction
                    )

                if re.search(r"\bamplification\b", structured_intent_text):
                    return (
                        "STRUCTURED DETAIL: Return only the Amplification block "
                        "from the requested Rule or Directive."
                        + target_instruction
                    )

                if re.search(r"\brationale\b", structured_intent_text):
                    return (
                        "STRUCTURED DETAIL: Return only the Rationale block "
                        "from the requested Rule or Directive."
                        + target_instruction
                    )

                if re.search(r"\bexamples?\b", structured_intent_text):
                    return (
                        "STRUCTURED DETAIL: Return only the Example or Examples "
                        "block from the requested Rule or Directive."
                        + target_instruction
                    )

                if re.search(r"\bexceptions?\b", structured_intent_text):
                    return (
                        "STRUCTURED DETAIL: Return only the Exception or Exceptions "
                        "block from the requested Rule or Directive."
                        + target_instruction
                    )

                if re.search(r"\bsee\s+also\b", structured_intent_text):
                    return (
                        "STRUCTURED DETAIL: Return only the See also block "
                        "from the requested Rule or Directive."
                        + target_instruction
                    )

            if (
                structured_reference.kind in {"rule", "directive"}
                and (
                    re.search(
                        r"^what\s+does\b.+\bsay\b",
                        clean
                    )
                    or re.search(
                        r"^what\s+is\s+(?:rule|dir(?:ective)?)\b",
                        clean
                    )
                    or re.search(
                        r"^(?:state|give|provide)\b.*\b(?:requirement|rule\s+statement|directive\s+statement)\b",
                        clean
                    )
                    or re.search(
                        r"\b(?:requirement|rule\s+statement|directive\s+statement)\b.*\b(?:rule|directive|dir)\s+\d",
                        clean
                    )
                    or re.fullmatch(
                        r"(?:misra\s+)?(?:rule|dir(?:ective)?)\s+\d+(?:\.\d+)*(?:[a-z])?\s*[?.!]*",
                        clean,
                        flags=re.IGNORECASE,
                    )
                )
            ):

                return (
                    "STRUCTURED STATEMENT: Return the exact requested Rule "
                    "or Directive statement itself. The first direct "
                    "requirement/title line has priority over rationale, "
                    "examples, cross-references, or applicability details."
                    + target_instruction
                )

        # Two-event temporal comparisons must be recognized before the generic
        # compound-question detector.  Otherwise a phrasing variant can fall
        # through to normal generation and return only one of the two dates.
        if self._is_explicit_temporal_comparison_question(question):

            return (
                "COMPARISON: Compare the two explicitly requested events using "
                "only grounded company-knowledge dates. State which came first "
                "and which came later."
                + target_instruction
            )

        if self._is_compound_question(
            question
        ):

            return (
                "COMPOUND: Answer every requested part in the original order. "
                "Do not omit a clause. Keep each part explicitly grounded."
                + target_instruction
            )

        # Explicit before/after comparisons are comparisons, not single-date
        # lookups. Treat Tagalog ``nauna ... kaysa ...`` and equivalent
        # English forms as two-fact requests so retrieval and deterministic
        # answer selection preserve both events.
        if self._is_explicit_temporal_comparison_question(clean):
            return (
                "COMPARISON: Compare the two explicitly requested events using "
                "only grounded company-knowledge dates. State which came first "
                "and which came later."
                + target_instruction
            )

        # Explicit comparison intent must win over individual quantity/date
        # words that happen to appear inside the compared items.
        if (
            re.search(r"\b(?:compare|comparison|contrast|versus|vs\.?|difference|different)\b", clean)
            or re.search(r"\b(?:ihambing|ikumpara|paghambingin|pagkakaiba)\b", clean)
            or re.search(r"\b(?:compare|comparison|contrast|versus|vs\.?|difference|different)\b", resolved_clean.lower())
        ):
            return (
                "COMPARISON: Compare every explicitly requested item using only "
                "supported company-knowledge facts. State the corresponding "
                "values or differences for each item and do not collapse the "
                "comparison into a single neighboring fact."
                + target_instruction
            )

        # Explicit explanation/summary intent must win over nouns such as
        # ``approval`` or ``eligibility`` that appear inside the requested
        # summary. Otherwise a multi-point policy summary can be misclassified
        # as a narrow APPROVER lookup.
        if (
            re.search(
                r"^(?:explain|describe|summari[sz]e|give\s+(?:me\s+)?(?:a\s+)?summary|"
                r"provide\s+(?:a\s+)?summary|ipaliwanag|ilarawan|ibuod)\b",
                clean,
                re.IGNORECASE,
            )
            or re.search(
                r"\b(?:short\s+summary|overall\s+summary|summary\s+of|overview\s+of)\b",
                clean,
                re.IGNORECASE,
            )
        ):
            return (
                "GROUNDED EXPLANATION: Explain or summarize the requested subject "
                "using all explicitly requested supported points from COMPANY "
                "KNOWLEDGE. Do not collapse a multi-point summary into one nearby "
                "field merely because words such as approval or eligibility are "
                "present. Prefer 2 to 5 concise sentences or a short bullet list, "
                "and do not add outside knowledge."
                + target_instruction
            )

        # Explicit question form has priority over words appearing later.
        if (
            re.search(
                r"^how\s+(?:many|much)\b",
                clean
            )
            or re.search(
                r"^how\s+(?:many|much)\b",
                resolved_clean.lower()
            )
            or re.search(
                r"\b(?:annual|annually|yearly|per\s+year)\b.*\b(?:allotment|allowance|entitlement|credits?|days?|hours?|amount|limit)\b",
                clean
            )
            or re.search(
                r"\b(?:annual|annually|yearly|per\s+year)\b.*\b(?:allotment|allowance|entitlement|credits?|days?|hours?|amount|limit)\b",
                resolved_clean.lower()
            )
        ):

            return (
                "QUANTITY: Return the requested number, amount, duration, "
                "size, limit, count, or value with its unit when available."
                + target_instruction
            )

        # Natural "who can use these benefits?" wording is an eligibility
        # relation when the user names a policy/benefit, not generic system
        # authorization.  Recognize English and Tagalog forms before the broad
        # AUTHORIZED ENTITY fallback.
        if re.search(
            r"\b(?:who\s+(?:can|may)\s+use|"
            r"sino\s+(?:ang\s+)?(?:pwede|puwede|pwedeng|puwedeng|maaaring)\s+(?:gumamit|makagamit))\b",
            clean,
            re.IGNORECASE,
        ):
            return (
                "ELIGIBLE OR ENTITLED ENTITY: Return the person, group, role, "
                "category, organization, system, component, or entity that is "
                "eligible, qualified, or entitled. Do not substitute a quantity, "
                "time, location, reason, approver, or procedure."
                + target_instruction
            )

        # An explicit eligibility/entitlement request in the *current user
        # turn* must outrank relation words carried only by a grounded anchor.
        # This prevents a follow-up such as "Who is actually eligible...?"
        # from being reclassified as APPROVER merely because the prior policy
        # summary also mentioned approval.
        if re.search(
            r"\b(?:eligible|eligibility|qualified|qualification|qualify|qualifies|covered|"
            r"entitled|entitlement)\b",
            clean,
        ):
            return (
                "ELIGIBLE OR ENTITLED ENTITY: Return the person, group, role, "
                "category, organization, system, component, or entity that is "
                "eligible, qualified, or entitled. Do not substitute a "
                "quantity, time, location, reason, approver, or procedure."
                + target_instruction
            )

        # Approval has priority over generic authorization.
        if (
            re.search(
                r"^(?:okay\s*[,;:]?\s*)?sino\s+(?:naman\s+)?(?:ang\s+)?(?:nag[- ]?aapprove|nag[- ]?approve|umaapprove|mag[- ]?approve|nag[- ]?aapruba|mag[- ]?apruba)\b",
                clean,
                re.IGNORECASE,
            )
            or re.search(
                r"^kanino\s+.*(?:ipa[- ]?approve|ipa[- ]?apruba)\b",
                clean,
                re.IGNORECASE,
            )
            or re.search(
                r"^who\s+(?:(?:can|may|must)\s+|(?:needs?|required)\s+to\s+)?"
                r"(?:approve|approves|approved|authorize|authorizes|authorized)\b",
                clean
            )
            or re.search(
                r"\b(?:approver|approvers|approval authority|approval authorities|approval required|requires approval)\b",
                clean
            )
            or re.search(
                r"^who\s+(?:(?:can|may|must)\s+|(?:needs?|required)\s+to\s+)?"
                r"(?:approve|approves|approved|authorize|authorizes|authorized)\b",
                resolved_clean.lower()
            )
            or re.search(
                r"\b(?:approval|approve|approver|authorize|authorization)\b",
                resolved_clean.lower()
            )
        ):

            return (
                "APPROVER: Return the approver, approving role, approval "
                "authority, or approving entity."
                + target_instruction
            )

        if (
            re.search(
                r"^who\s+(?:is|are|was|were)\s+responsible\b",
                clean
            )
            or re.search(
                r"^who\s+owns\b",
                clean
            )
            or re.search(
                r"\b(?:responsible|responsibility|owner|ownership)\b",
                clean
            )
        ):

            return (
                "RESPONSIBLE ENTITY: Return the person, group, role, team, "
                "organization, system, component, or entity responsible."
                + target_instruction
            )

        if (
            re.search(
                r"^who\s+(?:can|may)\b",
                clean
            )
            or re.search(
                r"^(?:who|which\s+(?:person|people|role|roles|team|group|organization|entity|entities))"
                r"\b.*\b(?:authorized|allowed|permitted)\b",
                clean,
            )
            or re.search(
                r"\b(?:authorized|authorization|permission)\b",
                clean
            )
        ):

            return (
                "AUTHORIZED ENTITY: Return the person, group, role, team, "
                "organization, system, component, or entity allowed to act."
                + target_instruction
            )

        if (
            re.search(
                r"\b(?:eligible|eligibility|qualified|qualification|qualify|qualifies|covered|"
                r"entitled|entitlement)\b",
                resolved_clean.lower()
            )
        ):

            return (
                "ELIGIBLE OR ENTITLED ENTITY: Return the person, group, role, "
                "category, organization, system, component, or entity that is "
                "eligible, qualified, or entitled. Do not substitute a "
                "quantity, time, location, reason, or procedure."
                + target_instruction
            )

        if (
            re.search(r"^when\b", clean)
            or re.search(r"^when\b", resolved_clean.lower())
        ):

            return (
                "TIME: Return the date, time, schedule, period, deadline, "
                "frequency, sequence point, or triggering condition."
                + target_instruction
                + " Do not substitute another date or time merely because "
                "it appears in the same context."
            )

        if (
            re.search(r"^where\b", clean)
            or re.search(r"^where\b", resolved_clean.lower())
        ):

            return (
                "LOCATION: Return the requested place, path, section, module, "
                "system area, storage location, interface, or position."
                + target_instruction
            )

        if (
            re.search(
                r"^what\s+(?:issue|matter|topic|question|decision)\b",
                clean
            )
            or re.search(
                r"^what\s+(?:issue|matter|topic|question|decision)\b",
                resolved_clean.lower()
            )
        ):
            return (
                "SUBJECT MATTER: Return the exact issue, matter, topic, question, "
                "or decision that the requested subject discussed, consulted on, "
                "considered, or sought advice about."
                + target_instruction
            )

        if (
            re.search(
                r"\b(?:purpose|objective|goal|aim|mission)\b",
                clean
            )
            or re.search(
                r"\b(?:purpose|objective|goal|aim|mission)\b",
                resolved_clean.lower()
            )
            or re.search(
                r"\b(?:layunin|adhikain|mithiin|hangarin)\b",
                clean
            )
        ):
            return (
                "PURPOSE: Return the explicitly stated purpose, objective, goal, "
                "aim, mission, or intended outcome for the requested subject."
                + target_instruction
            )

        if (
            re.search(
                r"\b(?:affect|affected|effect|effects|impact|impacted|mean\s+for|meant\s+for|result\s+for)\b",
                clean
            )
            or re.search(
                r"\b(?:affect|affected|effect|effects|impact|impacted|mean\s+for|meant\s+for|result\s+for)\b",
                resolved_clean.lower()
            )
        ):
            return (
                "EFFECT: Return the explicitly supported effect, consequence, "
                "outcome, or change for the exact target named in the question. "
                "Do not substitute an effect on a different nearby entity."
                + target_instruction
            )

        if (
            re.search(
                r"^(?:what|which)\s+(?:caused|causes)\b",
                clean
            )
            or re.search(
                r"^what\s+(?:is|was)\s+the\s+cause\b",
                clean
            )
            or re.search(
                r"^(?:ano|anong)\s+(?:ang\s+)?(?:sanhi|dahilan)\b",
                clean
            )
        ):

            return (
                "REASON: Return the explicitly stated cause. Give priority to "
                "a confirmed or conclusive cause over a rumor, allegation, "
                "superseded explanation, or debunked claim."
                + target_instruction
            )

        if re.search(
            r"^why\b",
            clean
        ):

            return (
                "REASON: Return the stated reason, rationale, purpose, "
                "cause, or justification."
                + target_instruction
            )

        if (
            re.search(
                r"^how\s+to\b",
                clean
            )
            or re.search(
                r"\b(?:steps|procedure|procedures|workflow|"
                r"installation|configuration|setup|troubleshooting)\b",
                clean
            )
        ):

            return (
                "PROCEDURE: Return the method or ordered steps. "
                "Use numbering when multiple steps are present."
                + target_instruction
            )

        # Plain identity questions should provide a useful overview.
        # Include common Tagalog identity forms so they receive the same
        # grounded identity safeguards as their English equivalents.
        if (
            re.search(
                r"^(?:who\s+(?:is|was)|sino\s+(?:si|ang))\b",
                clean
            )
            or re.search(
                r"^(?:who\s+(?:is|was)|sino\s+(?:si|ang))\b",
                resolved_clean.lower()
            )
            or re.search(r"\b(?:biography|profile|description)\b", resolved_clean.lower())
        ):

            return (
                "IDENTITY OR OVERVIEW: Identify the subject and provide a "
                "brief useful description of who or what it is, including "
                "its role, purpose, significance, or key details when "
                "explicitly supported. Do not return only the subject name "
                "unless no other relevant information is available."
                + target_instruction
            )

        if re.search(
            r"^(?:which person|which people|which role|which roles|"
            r"which team|which teams|which system|which systems|"
            r"which component|which components|which entity|which entities)\b",
            clean
        ):

            return (
                "PERSON OR ENTITY: Return the requested person, group, role, "
                "team, organization, system, component, category, or entity. "
                "Do not substitute another fact type."
                + target_instruction
            )

        if re.search(
            r"^which\b",
            clean
        ):

            return (
                "ENTITY OR CHOICE: Return the exact entity, item, group, "
                "option, category, system, component, role, or name that "
                "satisfies the relationship described in the question. "
                "Do not return a nearby but differently related entity."
                + target_instruction
            )

        # Direct relationship questions need a tighter entity focus than
        # GENERAL so a nearby metadata item (for example a newspaper title)
        # cannot substitute for the organization/role/title actually asked.
        if re.search(
            r"^(?:what|which)\s+(?:organization|organisation|group|association)\b",
            clean
        ) or re.search(
            r"^(?:ano|anong)\s+(?:ang\s+)?(?:organisasyon|samahan)\b",
            clean
        ):

            return (
                "ENTITY OR CHOICE: Return the exact organization, group, "
                "association, society, or movement that satisfies the "
                "relationship in the question. Do not substitute a newspaper, "
                "publication, office, title, person, or nearby entity."
                + target_instruction
            )

        if re.search(
            r"^(?:what|which)\s+(?:position|role|title|office)\b",
            clean
        ) or re.search(
            r"^(?:ano|anong)\s+(?:ang\s+)?(?:posisyon|tungkulin|titulo)\b",
            clean
        ):

            return (
                "ENTITY OR CHOICE: Return the exact position, role, title, "
                "or office explicitly connected to the requested subject and "
                "relationship. Include multiple simultaneously stated roles "
                "when the source explicitly gives them together."
                + target_instruction
            )

        if (
            re.search(
                r"^(?:what are|who are|list|enumerate|name the|"
                r"give me the list)\b",
                clean
            )
            or self._is_multi_answer_question(
                question
            )
            or self._looks_like_plural_list_question(
                question
            )
        ):

            return (
                "LIST: Return every relevant explicitly stated item. "
                "Use one Markdown bullet per item."
                + target_instruction
                + " Scan the complete retrieved context, not only the first "
                "paragraph or first list."
            )

        if re.search(
            r"^(?:explain|describe)\b",
            clean
        ):

            return (
                "GROUNDED EXPLANATION: Explain the requested subject using "
                "the important relevant information explicitly supported by "
                "COMPANY KNOWLEDGE. Cover what it is or requires, and include "
                "the relevant rationale, scope, conditions, actions, "
                "implications, or examples when those are present and help "
                "the user understand the topic. Prefer 2 to 5 concise "
                "sentences or a short bullet list for several distinct "
                "points. Do not force missing categories, add outside "
                "knowledge, or pad the answer with unrelated details."
                + target_instruction
            )

        if re.search(
            r"^(?:what is|what was|define|ano\s+ang|anong\s+)\b",
            clean
        ):

            return (
                "DEFINITION OR DETAIL: Return the direct definition, rule, "
                "requirement, behavior, configuration, or requested detail."
                + target_instruction
            )

        polite_request = re.search(
            r"^(?:can|could|would|will)\s+you\b",
            clean
        )

        if (
            not polite_request
            and re.search(
                r"^(?:is|are|was|were|does|do|did|can|could|"
                r"should|must|has|have|had)\b",
                clean
            )
        ):

            return (
                "YES OR NO: Start with Yes or No only when explicitly "
                "supported, then add one brief supporting statement."
                + target_instruction
            )

        # Short names, titles, standards, rules, commands, codes,
        # policies, systems, and topic phrases receive a useful overview.
        topic_words = re.findall(
            r"[A-Za-z0-9À-ÖØ-öø-ÿ_'’-]+",
            clean
        )

        if (
            1 <= len(topic_words) <= 6
            and not re.search(
                r"\b(?:who|what|when|where|why|how|is|are|was|were|"
                r"does|do|did|can|could|should|must|has|have|had)\b",
                clean
            )
        ):

            return (
                "SHORT TOPIC OVERVIEW: Provide a brief useful overview of "
                "the topic, name, title, standard, rule, command, code, "
                "policy, system, or phrase. Include its role, purpose, "
                "meaning, or key details when explicitly supported. Do not "
                "return only the title or name unless no other relevant "
                "information is available."
                + target_instruction
            )

        return (
            "GENERAL: Return only the directly requested information. "
            "Do not add unrelated details."
            + target_instruction
        )


    def _normalize_identity_match_text(self, text: str):

        if not text:
            return ""

        normalized = unicodedata.normalize(
            "NFKD",
            str(text)
        )

        normalized = "".join(
            character
            for character in normalized
            if not unicodedata.combining(character)
        )

        normalized = normalized.casefold()
        normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            normalized
        )

        return re.sub(
            r"\s+",
            " ",
            normalized
        ).strip()

    def _identity_subject_tokens(
        self,
        question: str,
        resolved_question: str = ""
    ):

        target = self._normalize_identity_match_text(
            resolved_question or question
        )

        ignored = {
            "who", "is", "was", "are", "were", "what", "about",
            "sino", "si", "ang",
            "tell", "me", "explain", "describe", "please", "the",
            "a", "an", "biography", "profile", "description", "life", "overview", "summary",
            "background", "known", "for", "important", "facts",
        }

        return [
            token
            for token in target.split()
            if len(token) >= 2
            and token not in ignored
        ]

    def _identity_subject_display(
        self,
        question: str,
        resolved_question: str = "",
    ):

        """Return a clean user-visible identity subject without prompt words."""

        candidates = (
            str(question or "").strip(),
            str(resolved_question or "").strip(),
        )

        patterns = (
            r"(?i)^who\s+(?:is|was)\s+(.+?)[?!.]*$",
            r"(?i)^sino\s+(?:si|ang)\s+(.+?)[?!.]*$",
            r"(?i)^give\s+(?:me\s+)?(?:a\s+)?(?:short\s+|brief\s+|concise\s+)?(?:profile|description)\s+of\s+(.+?)(?:\s+from\s+(?:the\s+)?(?:knowledge\s+base|company\s+knowledge|stored\s+material|stored\s+documents|documents))?[?!.]*$",
            r"(?i)^ano\s+ang\s+(.+?)\s+sa\s+(?:maikling|simpleng)\s+paliwanag[?!.]*$",
            r"(?i)^(.+?)\s+(?:biography|profile|description)$",
        )

        for candidate in candidates:
            candidate = re.sub(r"\s+", " ", candidate)
            for pattern in patterns:
                match = re.match(pattern, candidate)
                if match:
                    subject = match.group(1).strip(" ,.;:?")
                    if subject:
                        return subject

        return ""

    def _clean_grounded_identity_sentence(
        self,
        sentence: str,
        question: str,
        resolved_question: str = "",
    ):

        """Remove PDF heading/pronunciation noise without changing the fact."""

        if not sentence:
            return ""

        cleaned = re.sub(
            r"(?i)\(\s*(?:tagalog|spanish)\s*:.*?\)\s*(?=(?:is|was|are|were)\b)",
            " ",
            sentence,
        )
        cleaned = re.sub(r"\[[^\]]{1,12}\]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        subject = self._identity_subject_display(
            question,
            resolved_question,
        )

        # PDF lead blocks often repeat a heading/name and insert IPA/date
        # metadata before the actual definitional predicate. When a direct
        # copular predicate exists, keep that source predicate but anchor it to
        # the subject explicitly named by the user.
        if subject:
            predicate = re.search(
                r"\b(?:is|was|are|were)\b\s+.+",
                cleaned,
                flags=re.IGNORECASE,
            )
            if predicate:
                predicate_text = predicate.group(0).strip()
                cleaned = f"{subject} {predicate_text}"

        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        return cleaned.strip()

    def _grounded_identity_overview_from_context(
        self,
        context: str,
        question: str,
        resolved_question: str = ""
    ):

        """Return one direct identity sentence already present in context.

        This is a deterministic recovery guard for the rare case where a
        verifier collapses a supported identity overview to only the subject
        name. It never introduces outside facts: the descriptive predicate is
        taken from accepted company context and PDF heading/pronunciation noise
        is removed deterministically.
        """

        if not context:
            return ""

        subject_tokens = self._identity_subject_tokens(
            question,
            resolved_question
        )

        if not subject_tokens:
            return ""

        clean_context = re.sub(
            r"(?m)^=====\s*DOCUMENT\s+\d+\s*=====\s*$",
            " ",
            context
        )

        clean_context = re.sub(
            r"\s+",
            " ",
            clean_context
        ).strip()

        # Keep sentence boundaries conservative so abbreviations and dates do
        # not create dozens of tiny fragments.
        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ0-9])",
            clean_context
        )

        best_sentence = ""
        best_score = -1.0

        for sentence in sentences:
            candidate = re.sub(
                r"\s+",
                " ",
                sentence
            ).strip()

            if not candidate:
                continue

            if len(candidate.split()) < 6:
                continue

            normalized_candidate = self._normalize_identity_match_text(
                candidate
            )

            matched_tokens = sum(
                1
                for token in subject_tokens
                if token in normalized_candidate
            )

            required_matches = min(
                2,
                len(subject_tokens)
            )

            if matched_tokens < required_matches:
                continue

            identity_signals = (
                "nationalist", "writer", "polymath", "engineer", "manager",
                "leader", "founder", "physician", "doctor", "hero",
                "organization", "company", "lawyer", "statesman",
                "president", "prime minister", "revolutionary", "educator",
            )
            has_definitional_predicate = bool(re.search(
                r"\b(?:is|was|are|were)\s+(?:a|an|the)\s+[a-z]",
                normalized_candidate
            ))
            if not (has_definitional_predicate and any(
                signal in normalized_candidate for signal in identity_signals
            )):
                continue

            # Avoid a citation/reference sentence even if it happens to
            # contain the subject name and a copular verb.
            if re.search(
                r"\b(?:citations|references|bibliography|external links|"
                r"isbn|doi|https|www)\b",
                normalized_candidate
            ):
                continue

            score = float(matched_tokens)

            score += sum(
                0.20
                for signal in identity_signals
                if signal in normalized_candidate
            )

            # Prefer concise definitional sentences instead of long tangents.
            if len(candidate) <= 500:
                score += 0.30

            if score > best_score:
                best_score = score
                best_sentence = candidate

        return self._clean_grounded_identity_sentence(
            best_sentence,
            question,
            resolved_question,
        )

    def _bm25_identity_fast_path_answer(
        self,
        context: str,
        results,
        question: str,
        resolved_question: str,
        answer_focus: str,
    ):

        """Return a deterministic identity sentence for the strict BM25 path.

        The retriever marks this path only after selecting a subject-bound
        introductory definition.  Do not ask the LLM to rewrite that evidence:
        extracting the supported sentence directly is both faster on low-spec
        PCs and prevents the model from adding outside facts.
        """

        if not (
            answer_focus
            and answer_focus.startswith("IDENTITY OR OVERVIEW:")
            and context
            and results
            and len(results) == 1
            and bool(results[0].get("_bm25_identity_fast_path"))
        ):
            return ""

        grounded = self._grounded_identity_overview_from_context(
            context=context,
            question=question,
            resolved_question=resolved_question,
        )

        if not grounded:
            return ""

        evidence_logger.record_event(
            event_name="BM25 IDENTITY GROUNDING",
            status="DETERMINISTIC SOURCE SENTENCE",
            details={
                "reason": (
                    "The strict BM25 identity fast path selected a direct "
                    "subject-bound definition, so the user-facing answer was "
                    "extracted from that accepted sentence without LLM "
                    "generation or outside-knowledge expansion."
                ),
                "source": results[0].get("metadata", {}).get(
                    "file_name", "Unknown"
                ),
                "chunk_id": results[0].get("metadata", {}).get(
                    "chunk_id", ""
                ),
            },
        )

        return grounded

    def _is_thin_identity_answer(self, answer: str):

        if not answer:
            return True

        clean = re.sub(
            r"\s+",
            " ",
            answer
        ).strip()

        if clean.lower() == NO_RESULT_MESSAGE.strip().lower():
            return True

        if len(clean.split()) > 10:
            return False

        normalized = self._normalize_identity_match_text(
            clean
        )

        return not re.search(
            r"\b(?:is|was|are|were)\b",
            normalized
        )

    def _preserve_grounded_identity_overview(
        self,
        context: str,
        question: str,
        resolved_question: str,
        answer_focus: str,
        verified_answer: str
    ):

        if not (
            answer_focus
            and answer_focus.startswith("IDENTITY OR OVERVIEW:")
        ):
            return verified_answer

        if not self._is_thin_identity_answer(
            verified_answer
        ):
            return verified_answer

        grounded_overview = self._grounded_identity_overview_from_context(
            context=context,
            question=question,
            resolved_question=resolved_question
        )

        if not grounded_overview:
            return verified_answer

        evidence_logger.record_event(
            event_name="IDENTITY OVERVIEW SAFETY",
            status="PRESERVED GROUNDED OVERVIEW",
            details={
                "reason": (
                    "Verifier output was name-only/thin while accepted company "
                    "context contained a direct supported identity sentence."
                )
            }
        )

        return grounded_overview

    def _identity_draft_can_skip_focus_verifier(
        self,
        draft_answer: str,
        question: str,
        resolved_question: str = "",
    ):

        """Conservatively skip a redundant identity verifier call.

        Only one concise, complete identity sentence qualifies. Longer or
        ambiguous drafts keep the established LLM verifier path.
        """

        if not draft_answer:
            return False

        clean = re.sub(r"\s+", " ", draft_answer).strip()

        if (
            clean.lower() == NO_RESULT_MESSAGE.strip().lower()
            or self._contains_prompt_leak(clean)
            or len(clean.split()) < 8
            or len(clean.split()) > 45
        ):
            return False

        # Restrict the bypass to one concise sentence. This intentionally
        # leaves richer/multi-sentence drafts on the existing verifier path.
        sentence_marks = len(re.findall(r"[.!?](?:\s|$)", clean))
        if sentence_marks > 1:
            return False

        normalized = self._normalize_identity_match_text(clean)
        subject_tokens = self._identity_subject_tokens(
            question,
            resolved_question
        )

        if not subject_tokens:
            return False

        required_matches = min(2, len(subject_tokens))
        matched = sum(
            1
            for token in subject_tokens
            if token in normalized
        )

        if matched < required_matches:
            return False

        if not re.search(r"\b(?:is|was|are|were)\b", normalized):
            return False

        if re.match(r"^(?:yes|no)\b", normalized):
            return False

        # Require meaningful descriptive content beyond the subject itself.
        ignored = set(subject_tokens) | {
            "is", "was", "are", "were", "a", "an", "the", "and",
            "of", "in", "on", "for", "to", "from", "with", "who",
        }
        descriptive = [
            token
            for token in normalized.split()
            if len(token) >= 3 and token not in ignored
        ]

        return len(descriptive) >= 3

    def _compound_facets_for_answer_check(self, question: str):

        if not question:
            return []

        clean = re.sub(r"\s+", " ", question.lower().strip())
        clean = re.sub(
            r"^(?:explain|describe|summari[sz]e|ipaliwanag|ilarawan|ibuod)\s+",
            "",
            clean,
            flags=re.IGNORECASE,
        )
        interrogative = (
            r"(?:who|what|when|where|why|how|which|"
            r"sino|ano|anong|kailan|saan|bakit|paano|alin)"
        )
        imperative = (
            r"(?:explain|describe|summari[sz]e|define|list|enumerate|"
            r"give|show|provide|tell|clarify|elaborate|"
            r"ipaliwanag|ilarawan|ibuod|ilista|ibigay)"
        )
        request_starter = rf"(?:{interrogative}|{imperative})"

        parts = re.split(
            rf"(?:\b(?:and|at)\b\s+(?:also\s+|din\s+|rin\s+)?"
            rf"(?={request_starter}\b)|"
            rf"[?;]\s*(?:please\s+|paki\s*)?(?={request_starter}\b))",
            clean,
            flags=re.IGNORECASE,
        )

        facets = [part.strip(" ,.;?") for part in parts if part.strip(" ,.;?")]
        return facets if len(facets) >= 2 else []

    def _compound_draft_can_skip_verifier(
        self,
        question: str,
        draft_answer: str,
    ):

        """Return True when every explicit compound facet is visibly covered.

        The gate stays conservative, but it understands imperative second
        clauses such as ``explain it`` and ``explain why ...``. This avoids a
        second LLM pass when a grounded draft already satisfies the request.
        """

        if not draft_answer:
            return False

        clean_answer = re.sub(r"\s+", " ", draft_answer).strip()
        normalized = self._normalize_identity_match_text(clean_answer)

        if (
            clean_answer.lower() == NO_RESULT_MESSAGE.strip().lower()
            or self._contains_prompt_leak(clean_answer)
        ):
            return False

        facets = self._compound_facets_for_answer_check(question)
        if not facets or len(facets) > 4:
            return False

        anchor_tokens = [
            token
            for token in self._normalize_identity_match_text(facets[0]).split()
            if len(token) >= 3
            and token not in {
                "who", "what", "when", "where", "why", "how", "which",
                "sino", "ano", "anong", "kailan", "saan", "bakit",
                "paano", "alin", "was", "were", "is", "are", "did",
                "does", "do", "the", "and", "at", "explain",
                "describe", "summarize", "summarise", "ipaliwanag",
            }
        ][:4]

        reason_signals = (
            "because", "due", "reason", "rationale", "purpose",
            "important", "significant", "inspired", "influence",
            "influential", "led", "resulted", "enabled", "helped",
            "prevent", "protect", "ensure", "risk", "impact",
            "dahil", "sapagkat", "mahalaga", "layunin", "sanhi",
        )

        date_re = re.compile(
            r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
            r"\d{4}|(?:january|february|march|april|may|june|july|"
            r"august|september|october|november|december)\s+\d{1,2})\b",
            re.IGNORECASE,
        )

        imperative_words = {
            "explain", "describe", "summarize", "summarise", "define",
            "clarify", "elaborate", "ipaliwanag", "ilarawan", "ibuod",
        }

        for facet in facets:
            facet_norm = self._normalize_identity_match_text(facet)
            facet_words = facet_norm.split()

            if not facet_words:
                return False

            first = facet_words[0]

            # Imperative clauses may wrap an interrogative relation, e.g.
            # ``explain why it was important``. Evaluate the embedded relation
            # rather than treating ``explain`` as an unknown facet type.
            if first in imperative_words:
                embedded = facet_words[1] if len(facet_words) > 1 else ""

                if embedded in {"why", "bakit"}:
                    if not any(signal in normalized for signal in reason_signals):
                        return False
                    continue

                # Generic explain/describe request: require a substantive
                # answer and visible connection to the first-clause subject.
                if len(normalized.split()) < 12:
                    return False
                if anchor_tokens and not any(
                    token in normalized for token in anchor_tokens
                ):
                    return False
                continue

            if first in {"who", "sino"}:
                if not re.search(r"\b(?:is|was|are|were)\b", normalized):
                    return False
                if anchor_tokens and not any(token in normalized for token in anchor_tokens):
                    return False
                continue

            if first in {"why", "bakit"}:
                if not any(signal in normalized for signal in reason_signals):
                    return False
                continue

            if first in {"when", "kailan"}:
                if not date_re.search(clean_answer):
                    return False
                continue

            if first in {"how", "paano"} and re.search(
                r"\b(?:many|much|ilan)\b", facet_norm
            ):
                if not re.search(r"\b\d+(?:\.\d+)?\b", clean_answer):
                    return False
                continue

            if first in {"what", "ano", "anong"}:
                # Definition/overview answers often use verbs such as
                # "explains" or "describes" instead of a copula.
                if not re.search(
                    r"\b(?:is|was|are|were|means|refers|requires|provides|has|"
                    r"explains|describes|defines|introduces|covers)\b",
                    normalized,
                ):
                    return False
                if anchor_tokens and not any(token in normalized for token in anchor_tokens):
                    return False
                continue

            # WHERE / WHICH / open HOW and other ambiguous clauses keep the
            # verifier because deterministic coverage is uncertain.
            return False

        return True

    def _question_subject_display(
        self,
        question: str,
    ):

        """Extract the explicitly named subject for narrow relation questions."""

        if not question:
            return ""

        clean = re.sub(r"\s+", " ", str(question)).strip()

        patterns = (
            r"(?i)^what\s+caused\s+(.+?)[’']s\s+.+?[?!.]*$",
            r"(?i)^what\s+(?:organization|organisation|group|association)\s+did\s+(.+?)\s+(?:co[-\s]?found|help(?:ed)?\s+found|found)[?!.]*$",
            r"(?i)^what\s+(?:position|role|office)\s+did\s+(.+?)\s+(?:hold|have|serve)(?:\s+|[?!.]|$)",
            r"(?i)^(?:what|which)\s+.+?\b(?:position|role|title|office)\b.+?\b(?:associated|connected)\s+with\s+(.+?)[?!.]*$",
            r"(?i)^what\s+title\s+is\s+(.+?)\s+often\s+called(?:\s+.*)?[?!.]*$",
        )

        for pattern in patterns:
            match = re.match(pattern, clean)
            if match:
                return match.group(1).strip(" ,.;:?")

        return ""

    @staticmethod
    def _clean_relation_value(value: str):

        if not value:
            return ""

        cleaned = re.sub(r"\[[^\]]{1,12}\]", "", str(value))
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")
        return cleaned

    @staticmethod
    def _question_role_scope_hint(question: str):
        """Extract a domain-neutral scope noun from a role/position question.

        Example: ``Which top Operations position is associated with Alex?``
        yields ``Operations``.  Generic modifiers are ignored; no document,
        person, company, or historical topic is encoded here.
        """
        if not question:
            return ""
        clean = re.sub(r"\s+", " ", str(question)).strip()
        match = re.match(
            r"(?i)^(?:what|which)\s+(.+?)\s+(?:position|role|title|office)\b",
            clean,
        )
        if not match:
            return ""
        words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9'’.-]+", match.group(1))
        generic = {
            "top", "main", "major", "primary", "highest", "key",
            "leadership", "senior", "official", "listed", "associated",
            "government", "company", "national", "current", "exact",
        }
        meaningful = [word for word in words if word.lower() not in generic]
        if not meaningful:
            return ""
        # The nearest noun phrase before role/position is the strongest scope.
        return " ".join(meaningful[-3:]).strip()

    def _grounded_direct_relation_answer(
        self,
        context: str,
        question: str,
        answer_focus: str,
    ):

        """Recover a narrow direct relation only from explicit accepted text.

        This guard is intentionally limited to strongly worded relation
        questions where the source text contains a direct lexical relation.
        It prevents a nearby metadata value from replacing the requested fact
        and can recover a supported answer when a small model returns fallback.
        """

        if not context or not question or not answer_focus:
            return ""

        clean_question = re.sub(
            r"\s+",
            " ",
            str(question).strip()
        )
        raw_context = re.sub(
            r"(?m)^=====\s*DOCUMENT\s+\d+\s*=====\s*$",
            "\n",
            str(context)
        )
        raw_lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in raw_context.splitlines()
            if re.sub(r"\s+", " ", line).strip()
        ]
        clean_context = re.sub(r"\s+", " ", raw_context).strip()

        subject = self._question_subject_display(clean_question)
        subject_words = self._normalize_identity_match_text(subject).split()
        surname = subject_words[-1] if subject_words else ""

        # Confirmed cause: prefer an explicit conclusive/confirmed cause over
        # rumor text. The Mabini corpus, for example, contains both a rumor
        # and a later autopsy conclusion.
        if (
            answer_focus.startswith("REASON:")
            and re.search(
                r"(?i)^what\s+(?:caused|causes)\b",
                clean_question
            )
        ):
            possessive = re.match(
                r"(?i)^what\s+caused\s+(.+?)[’']s\s+(.+?)[?!.]*$",
                clean_question
            )
            condition = (
                possessive.group(2).strip()
                if possessive
                else ""
            )

            cause_patterns = (
                r"(?i)\b(?:proved|confirmed|established)\b.{0,180}?"
                r"\bcause\s+of\s+(?:his|her|its|the)\s+"
                + (re.escape(condition) if condition else r"[a-z][a-z \-]{1,60}")
                + r"\s+was\s+([^.;]{1,100})",
                (
                    r"(?i)\b"
                    + (re.escape(condition) if condition else r"[a-z][a-z \-]{1,60}")
                    + r"\s+was\s+caused\s+by\s+([^.;]{1,100})"
                ),
            )

            for pattern in cause_patterns:
                match = re.search(pattern, clean_context)
                if not match:
                    continue

                cause = self._clean_relation_value(match.group(1))
                if not cause:
                    continue

                if re.search(
                    r"(?i)\b(?:rumou?r|alleged|claimed|speculated)\b",
                    cause
                ):
                    continue

                if subject and condition:
                    return (
                        f"{subject}'s {condition} was caused by {cause}."
                    )

                return f"The confirmed cause was {cause}."

        # Organization co-founder relation. Use a common-name statement tied
        # to the same accepted passage rather than a nearby publication field.
        if (
            answer_focus.startswith("ENTITY OR CHOICE:")
            and re.search(
                r"(?i)^what\s+(?:organization|organisation|group|association)\s+did\b.+\b(?:co[-\s]?found|help(?:ed)?\s+found|found)\b",
                clean_question
            )
        ):
            cofound_windows = re.finditer(
                r"(?i)\b(?:co[-\s]?founder|co[-\s]?founded|founded)\b.{0,520}",
                clean_context
            )

            for window_match in cofound_windows:
                window = window_match.group(0)

                if surname and surname not in self._normalize_identity_match_text(
                    clean_context[
                        max(0, window_match.start() - 220):
                        window_match.end()
                    ]
                ):
                    continue

                common_name = re.search(
                    r'(?i)\b(?:more\s+commonly\s+)?known\s+as\s+(?:the\s+)?["“]([^"”]+)["”]',
                    window
                )

                if common_name:
                    organization = self._clean_relation_value(
                        common_name.group(1)
                    )
                    if organization:
                        return organization

            # Fallback to a definitional organization sentence that explicitly
            # lists the named subject among its founders.
            if surname:
                definition = re.search(
                    r"(?i)\bthe\s+([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’ -]{1,80}?)"
                    r"\s+(?:is|was)\s+(?:(?:an?|the)\s+)?"
                    r".{0,100}?\b(?:organization|organisation|association|movement)\b"
                    r".{0,260}?\bfounded\b.{0,260}?\b"
                    + re.escape(surname)
                    + r"\b",
                    clean_context,
                )
                if definition:
                    organization = self._clean_relation_value(
                        definition.group(1)
                    )
                    if organization:
                        return organization

        # Generic role/position association.  Only accept candidates that are
        # explicitly tied to the named subject in the accepted source text.
        # Nearby nicknames or roles belonging to another person must never be
        # borrowed merely because the same scope word appears close by.
        if (
            answer_focus.startswith("ENTITY OR CHOICE:")
            and re.search(
                r"(?i)^(?:what|which)\s+.+?\b(?:position|role|title|office)\b"
                r".+?\b(?:associated|connected)\s+with\b",
                clean_question,
            )
            and surname
        ):
            scope_hint = self._question_role_scope_hint(clean_question)
            scope_norm = self._normalize_identity_match_text(scope_hint)
            wants_top_role = bool(re.search(
                r"(?i)\b(?:top|highest|supreme|main|primary|chief|leading)\b",
                clean_question,
            ))
            apex_terms = {
                "supreme", "president", "chair", "chairman", "chairperson",
                "chief", "head", "director", "leader", "lead", "principal",
            }

            relation_patterns = (
                # ``Surname was named/elected/appointed President``.
                r"(?i)\b" + re.escape(surname)
                + r"\b.{0,45}?\b(?:was\s+)?(?:named|elected|appointed|installed)"
                  r"\s+(?:as\s+)?(?:the\s+)?"
                  r"([A-Za-zÀ-ÖØ-öø-ÿ'’.-]+(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'’.-]+){0,3}?)"
                  r"(?=\s*(?:,|;|\.|\band\b|$))",
                # ``installed Surname as President``.
                r"(?i)\b(?:named|elected|appointed|installed)\b.{0,70}?\b"
                + re.escape(surname)
                + r"\b\s+as\s+(?:the\s+)?"
                  r"([A-Za-zÀ-ÖØ-öø-ÿ'’.-]+(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'’.-]+){0,3}?)"
                  r"(?=\s*(?:,|;|\.|\band\b|$))",
                # Common administration/table prose: ``Surname as President``.
                r"(?i)\b" + re.escape(surname)
                + r"\b\s+as\s+(?:the\s+)?"
                  r"([A-Za-zÀ-ÖØ-öø-ÿ'’.-]+(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'’.-]+){0,3}?)"
                  r"(?=\s*(?:,|;|\.|\band\b|$))",
                # ``Surname served as / became President``.
                r"(?i)\b" + re.escape(surname)
                + r"\b.{0,45}?\b(?:served\s+as|became)\s+(?:the\s+)?"
                  r"([A-Za-zÀ-ÖØ-öø-ÿ'’.-]+(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'’.-]+){0,3}?)"
                  r"(?=\s*(?:,|;|\.|\band\b|$))",
            )

            candidates = []
            for pattern in relation_patterns:
                for role_match in re.finditer(pattern, clean_context):
                    role = self._clean_relation_value(role_match.group(1))
                    if not role:
                        continue
                    role_norm = self._normalize_identity_match_text(role)
                    if not role_norm or role_norm == surname:
                        continue

                    # Keep the relation local to the subject and reward an
                    # explicit query scope when that scope is present nearby.
                    local = clean_context[
                        max(0, role_match.start() - 180):role_match.end() + 220
                    ]
                    local_norm = self._normalize_identity_match_text(local)
                    score = 10.0
                    if scope_norm and scope_norm in local_norm:
                        score += 2.0
                    if wants_top_role and set(role_norm.split()).intersection(apex_terms):
                        score += 5.0
                    if len(role_norm.split()) <= 3:
                        score += 0.25
                    candidates.append((score, role_match.start(), role))

            # Preserve compact table/infobox structure where the role label
            # appears *before* the person value, for example::
            #
            #     Supreme President
            #     Deodato Arellano (...)
            #     Roman Basa (...)
            #     Andres Bonifacio (...)
            #
            # This is a generic label/value recovery, not a document-specific
            # shortcut.  A candidate label must be short and role-like; for a
            # top/highest question it must also carry an apex-role signal.
            role_label_terms = apex_terms.union({
                "manager", "supervisor", "secretary", "treasurer", "fiscal",
                "comptroller", "minister", "governor", "administrator",
                "coordinator", "officer", "captain", "commander", "warden",
            })
            for line_index, line in enumerate(raw_lines):
                line_norm = self._normalize_identity_match_text(line)
                if not re.search(rf"\b{re.escape(surname)}\b", line_norm):
                    continue

                # Look only within the immediately preceding compact block and
                # stop at obvious paragraph-like prose. This prevents a role
                # belonging to a distant person from leaking across sections.
                for distance in range(1, min(9, line_index + 1)):
                    label = raw_lines[line_index - distance]
                    label_norm = self._normalize_identity_match_text(label)
                    label_words = label_norm.split()
                    if not label_words:
                        continue
                    if len(label_words) > 5 or len(label) > 90:
                        # Long prose is a structural boundary, not a label.
                        break
                    if surname in label_norm:
                        continue
                    term_hits = set(label_words).intersection(role_label_terms)
                    if not term_hits:
                        continue
                    if wants_top_role and not set(label_words).intersection(apex_terms):
                        continue

                    score = 11.0 + max(0.0, 1.5 - 0.15 * distance)
                    if wants_top_role:
                        score += 4.0
                    if scope_norm:
                        nearby = " ".join(
                            raw_lines[max(0, line_index - 10): min(len(raw_lines), line_index + 4)]
                        )
                        nearby_norm = self._normalize_identity_match_text(nearby)
                        if scope_norm in nearby_norm:
                            score += 1.5
                    candidates.append((score, line_index - distance, label))
                    # The nearest qualifying structural label is sufficient for
                    # this subject block; do not collect older labels above it.
                    break

            # Same-line compact labels such as ``President: Alex Rivera`` or
            # ``Supreme President - Alex Rivera`` are also explicit subject
            # relations and are safe to evaluate deterministically.
            same_line_pattern = re.compile(
                r"(?i)([A-Za-zÀ-ÖØ-öø-ÿ'’.-]+(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'’.-]+){0,3}?)"
                r"\s*(?:[:\-–—])\s*[^.;\n]{0,80}?\b" + re.escape(surname) + r"\b"
            )
            for line_index, line in enumerate(raw_lines):
                match = same_line_pattern.search(line)
                if not match:
                    continue
                role = self._clean_relation_value(match.group(1))
                role_norm = self._normalize_identity_match_text(role)
                role_words = role_norm.split()
                if not set(role_words).intersection(role_label_terms):
                    continue
                if wants_top_role and not set(role_words).intersection(apex_terms):
                    continue
                score = 13.0 + (4.0 if wants_top_role else 0.0)
                candidates.append((score, line_index, role))

            # Subject-led profile/member sentence, e.g.
            # ``Alex Rivera (...) – the third Supreme President (...) of X``.
            if scope_hint:
                member_pattern = (
                    r"(?i)\b" + re.escape(surname)
                    + r"\b(?:\s*\([^)]{0,100}\))?\s*[-–—:]\s*"
                      r"(?:the\s+)?(?:leading\s+|senior\s+|principal\s+)?"
                      r"(?:founder\s+and\s+the\s+)?"
                      r"(?:(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+)?"
                      r"([A-Za-zÀ-ÖØ-öø-ÿ'’.-]+(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'’.-]+){0,3}?)"
                      r"\s*\([^)]{1,180}\)\s+of\s+(?:the\s+)?[^.;]{0,80}?\b"
                    + re.escape(scope_hint) + r"\b"
                )
                for role_match in re.finditer(member_pattern, clean_context):
                    role = self._clean_relation_value(role_match.group(1))
                    role_norm = self._normalize_identity_match_text(role)
                    if not role_norm:
                        continue
                    score = 12.0
                    if wants_top_role and set(role_norm.split()).intersection(apex_terms):
                        score += 5.0
                    candidates.append((score, role_match.start(), role))

                # Immediate pronoun continuation is also explicitly
                # subject-bound: the pronoun sentence must directly follow a
                # sentence that names the requested subject.  This supports
                # profile prose without allowing roles from a later person.
                pronoun_pattern = (
                    r"(?i)\b" + re.escape(surname)
                    + r"\b[^.]{0,240}\.\s*(?:he|she|they)\s+"
                      r"(?:was|is|were|are)\s+[^.;]{0,100}?"
                      r"(?:and\s+later\s+)?"
                      r"([A-Za-zÀ-ÖØ-öø-ÿ'’.-]+(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'’.-]+){0,3}?)"
                      r"\s*\([^)]{1,180}\)\s+of\s+[^.;]{0,180}?\b"
                    + re.escape(scope_hint) + r"\b"
                )
                for role_match in re.finditer(pronoun_pattern, clean_context):
                    role = self._clean_relation_value(role_match.group(1))
                    role_norm = self._normalize_identity_match_text(role)
                    if not role_norm:
                        continue
                    score = 12.0
                    if wants_top_role and set(role_norm.split()).intersection(apex_terms):
                        score += 5.0
                    candidates.append((score, role_match.start(), role))

            if candidates:
                candidates.sort(key=lambda row: (-row[0], row[1]))
                return candidates[0][2]

        # Position/role relation. Prefer an explicit appointment statement
        # and preserve multiple roles when the same source states them together.
        if (
            answer_focus.startswith("ENTITY OR CHOICE:")
            and re.search(
                r"(?i)^what\s+(?:position|role|office)\s+did\b",
                clean_question
            )
            and surname
        ):
            appointment = re.search(
                r"(?i)\b"
                + re.escape(surname)
                + r"\b.{0,80}?\bwas\s+appointed\s+([^.;]{2,140})",
                clean_context
            )

            if appointment:
                roles = self._clean_relation_value(
                    appointment.group(1)
                )
                if roles:
                    return f"{subject} was appointed {roles}."

        # "What title is X often called?" has an explicit lexical relation.
        if (
            answer_focus.startswith("ENTITY OR CHOICE:")
            and re.search(
                r"(?i)^what\s+title\s+is\b.+\boften\s+called\b",
                clean_question
            )
        ):
            title_match = re.search(
                r'(?i)\b(?:he|she|they|'
                + (re.escape(surname) if surname else r"subject")
                + r')\s+(?:is|was|are|were)\s+often\s+called\s+["“]([^"”]+)["”]',
                clean_context
            )
            if title_match:
                title = self._clean_relation_value(title_match.group(1))
                if title:
                    return title

        return ""

    @staticmethod
    def _relation_probe_tokens(value: str):
        """Return conservative lexical stems for labeled relation matching.

        This intentionally avoids domain dictionaries.  It is used only to bind
        an explicit key/value-style source label to wording already present in
        the user's question.
        """
        stop = {
            "who", "what", "which", "is", "are", "was", "were", "the",
            "a", "an", "in", "on", "at", "of", "for", "to", "from",
            "under", "according", "temporary", "validation", "note",
            "role", "person", "people", "entity", "team", "group",
            "owner", "ownership", "owns", "own", "responsible",
            "responsibility", "accountable", "assigned", "assignee",
        }

        def stem(token: str):
            token = token.casefold()
            if token.endswith("ies") and len(token) > 5:
                return token[:-3] + "y"
            if token.endswith("es") and len(token) > 5:
                return token[:-2]
            if token.endswith("s") and len(token) > 4:
                return token[:-1]
            return token

        tokens = []
        for token in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", str(value or "")):
            lowered = token.casefold()
            if len(lowered) <= 2 or lowered in stop:
                continue
            tokens.append(stem(lowered))
        return set(tokens)

    def _grounded_labeled_responsible_entity_answer(
        self,
        context: str,
        question: str,
        answer_focus: str,
    ):
        """Return a source-exact responsibility value from an explicit label.

        Example source grammar::

            Escalation owner role: Archive Coordination Lead X7Q0.

        The guard is deliberately narrow: the answer focus must be responsibility/
        ownership, the source line must be an explicit label/value pair, the label
        itself must carry an ownership/responsibility marker, and at least one
        non-generic lexical stem from the question must also occur in that label.
        This preserves opaque IDs/codes exactly without asking an LLM to rewrite
        them and does not encode any document, role, topic, or expected answer.
        """
        if (
            not context
            or not question
            or not answer_focus.startswith("RESPONSIBLE ENTITY:")
        ):
            return ""

        question_tokens = self._relation_probe_tokens(question)
        if not question_tokens:
            return ""

        ownership_pattern = re.compile(
            r"(?i)\b(?:owner|ownership|responsible|responsibility|accountable|assignee)\b"
        )
        separator_pattern = re.compile(r"^(.{1,140}?)\s*(?::|\s[-–—]\s)\s*(.{1,240})$")

        candidates = []
        for index, raw_line in enumerate(str(context).splitlines()):
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line or line.startswith("====="):
                continue
            match = separator_pattern.match(line)
            if not match:
                continue
            label = match.group(1).strip()
            value = self._clean_relation_value(match.group(2))
            if not value or not ownership_pattern.search(label):
                continue

            label_tokens = self._relation_probe_tokens(label)
            overlap = question_tokens.intersection(label_tokens)
            if not overlap:
                continue

            # Prefer labels with more explicit lexical binding to the query,
            # then preserve source order for deterministic tie-breaking.
            candidates.append((len(overlap), -index, value))

        if not candidates:
            return ""

        candidates.sort(reverse=True)
        return candidates[0][2]

    def _grounded_approver_answer(
        self,
        context: str,
        question: str,
        answer_focus: str,
    ):
        """Return an explicitly source-bound approver without an LLM call.

        This is intentionally generic: it recognizes common approval grammar
        only inside accepted company context and binds candidates to lexical
        terms from the user's question.  It does not encode any document name,
        policy, employee role, or expected answer.
        """

        if (
            not context
            or not question
            or not answer_focus.startswith("APPROVER:")
        ):
            return ""

        normalized_question = self._normalize_identity_match_text(question)
        ignored = {
            "who", "what", "which", "must", "can", "may",
            "approve", "approves", "approved", "approval", "approver",
            "authorize", "authorized", "authorization", "an", "a",
            "the", "is", "are", "was", "were", "to", "of", "for",
            # Natural Tagalog/Taglish approval follow-ups often contain only
            # discourse markers plus the approval relation itself.  Treat
            # those words as non-scope tokens so an already grounded source
            # can answer directly instead of forcing an unnecessary LLM call.
            "sino", "kanino", "ang", "yung", "iyong", "naman", "okay",
            "ok", "nag", "mag", "ipa", "inaapprove", "aapprove",
        }
        question_tokens = {
            token
            for token in normalized_question.split()
            if len(token) >= 3 and token not in ignored
        }

        candidate_pattern = r"[A-Za-z][A-Za-z0-9/&'’ ._-]{0,80}?"
        relation_patterns = (
            re.compile(
                rf"(?i)\brequires?\s+(?:the\s+)?(?P<entity>{candidate_pattern})\s+approval\b"
            ),
            re.compile(
                rf"(?i)\brequired\s+(?:the\s+)?(?P<entity>{candidate_pattern})\s+approval\b"
            ),
            re.compile(
                rf"(?i)\b(?:must|shall|should)\s+be\s+(?:approved|authorized)\s+by\s+"
                rf"(?:the\s+)?(?P<entity>{candidate_pattern})(?=\s*(?:[.;,]|\band\b|$))"
            ),
            re.compile(
                rf"(?i)\bapproval\s+(?:is\s+)?(?:required\s+)?(?:from|by)\s+"
                rf"(?:the\s+)?(?P<entity>{candidate_pattern})(?=\s*(?:[.;,]|\band\b|$))"
            ),
            re.compile(
                rf"(?i)\b(?:approved|authorized)\s+by\s+(?:the\s+)?"
                rf"(?P<entity>{candidate_pattern})(?=\s*(?:[.;,]|\band\b|$))"
            ),
        )

        # Prefer compact source label/value pairs when available because they
        # preserve the local policy field that the relation belongs to.
        scopes = []
        for label, value in self._compact_label_value_pairs_from_context(context):
            combined = f"{label}\n{value}"
            if re.search(r"(?i)\b(?:approve|approved|approval|approver|authorize|authorized)\b", combined):
                scopes.append((combined, 3.0))

        # Also support ordinary prose sources that do not use field labels.
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", str(context)):
            clean_sentence = sentence.strip()
            if clean_sentence and re.search(
                r"(?i)\b(?:approve|approved|approval|approver|authorize|authorized)\b",
                clean_sentence,
            ):
                scopes.append((clean_sentence, 1.0))

        candidates = []
        for scope_index, (scope, base_score) in enumerate(scopes):
            normalized_scope = self._normalize_identity_match_text(scope)
            overlap = sum(
                1 for token in question_tokens
                if re.search(rf"\b{re.escape(token)}\b", normalized_scope)
            )
            # When the question carries relation-specific nouns (for example
            # leave/request), require at least one of them in the same source
            # scope so a different approval field cannot leak across the chunk.
            if question_tokens and overlap == 0:
                continue

            for pattern in relation_patterns:
                match = pattern.search(scope)
                if not match:
                    continue
                entity = re.sub(r"\s+", " ", match.group("entity")).strip(" .,:;-")
                entity = re.sub(r"(?i)^(?:the|a|an)\s+", "", entity).strip()
                if not entity or len(entity.split()) > 10:
                    continue
                # Reject captures that are themselves relation prose rather than
                # the approving person/role/entity.
                if re.search(
                    r"(?i)\b(?:request|requests|require|requires|required|approve|approved|approval)\b",
                    entity,
                ):
                    continue
                candidates.append((base_score + overlap, -scope_index, entity))

        if not candidates:
            return ""

        candidates.sort(reverse=True)
        return candidates[0][2]

    def _grounded_eligibility_answer(
        self,
        context: str,
        question: str,
        answer_focus: str,
    ):
        """Return an explicit eligibility/entitlement value from compact source facts.

        The method is intentionally label/value based and domain-neutral. It
        never infers that an approver is also an eligible entity merely because
        both facts occur in the same policy chunk.
        """

        if (
            not context
            or not question
            or not answer_focus.startswith("ELIGIBLE OR ENTITLED ENTITY:")
        ):
            return ""

        blocks = [
            block.strip()
            for block in re.split(
                r"(?m)^=====\s*DOCUMENT\s+\d+\s*=====\s*$",
                str(context),
            )
            if block.strip()
        ] or [str(context)]

        candidates = []
        global_index = 0
        for block in blocks:
            for label, value in self._compact_label_value_pairs_from_context(
                block,
                max_chars=4000,
            ):
                index = global_index
                global_index += 1
                label_norm = self._normalize_identity_match_text(label)
                value_norm = self._normalize_identity_match_text(value)
                score = 0

                if re.search(r"\b(?:eligibility|eligible|qualification|qualified)\b", label_norm):
                    score += 6
                if re.search(r"\b(?:eligibility|eligible|qualified|entitled|covered)\b", value_norm):
                    score += 4

                # A compact source value can state the eligible population without
                # repeating the literal word ``eligible``. Entitlement remains a
                # direct source relation; approval/approver language receives no
                # score here.
                if re.search(r"\bentitled\b", value_norm):
                    score += 3

                if score:
                    candidates.append((score, -index, re.sub(r"\s+", " ", value).strip()))

        if not candidates:
            return ""

        candidates.sort(reverse=True)
        return candidates[0][2]

    def _grounded_same_document_approval_eligibility_answer(
        self,
        context: str,
        question: str,
    ) -> str:
        """Compose approval and eligibility facts only when both are explicit.

        This closes a narrow same-document follow-up gap without treating an
        approver as the eligible population.  Both source relations must exist in
        the accepted context; otherwise the normal retrieval/generation path is
        left unchanged.
        """

        if not context or not question:
            return ""
        clean = self._normalize_identity_match_text(question)
        has_approval = bool(re.search(
            r"\b(?:approve|approves|approved|approval|approver)\b",
            clean,
        ))
        has_eligibility = bool(re.search(
            r"\b(?:eligible|eligibility|entitled|entitlement|qualified|qualification)\b",
            clean,
        ))
        relation_request = bool(re.search(
            r"\b(?:apply|applies|scope|same document|same source|based on|based sa|according to|under)\b",
            clean,
        ))
        if not (has_approval and has_eligibility and relation_request):
            return ""

        approval_candidates = []
        eligibility_candidates = []
        pair_index = 0
        for label, value in self._compact_label_value_pairs_from_context(
            context,
            max_chars=5000,
        ):
            label_norm = self._normalize_identity_match_text(label)
            value_norm = self._normalize_identity_match_text(value)
            combined = f"{label_norm} {value_norm}"
            cleaned_value = re.sub(r"\s+", " ", str(value)).strip()

            approval_score = 0
            if re.search(r"\b(?:approval|approver)\b", label_norm):
                approval_score += 10
            if re.search(r"\b(?:approve|approved|approval|approver)\b", value_norm):
                approval_score += 5
            if approval_score:
                approval_candidates.append(
                    (approval_score, -pair_index, cleaned_value)
                )

            # Prefer the explicit Eligibility/Qualification field over nearby
            # entitlement quantities such as "47 days vacation leave".  The
            # old first-match logic could select the first "entitled" sentence
            # in a policy chunk before ever reaching its Eligibility field.
            eligibility_score = 0
            if re.search(r"\b(?:eligibility|eligible|qualification|qualified)\b", label_norm):
                eligibility_score += 12
            if re.search(r"\b(?:eligible|qualified|covered)\b", value_norm):
                eligibility_score += 7
            if re.search(r"\bregular\s+employees?\b", value_norm):
                eligibility_score += 6
            if re.search(r"\bentitled\b", value_norm):
                eligibility_score += 2
            if eligibility_score:
                eligibility_candidates.append(
                    (eligibility_score, -pair_index, cleaned_value)
                )

            pair_index += 1

        approval_value = ""
        eligibility_value = ""
        if approval_candidates:
            approval_candidates.sort(reverse=True)
            approval_value = approval_candidates[0][2]
        if eligibility_candidates:
            eligibility_candidates.sort(reverse=True)
            eligibility_value = eligibility_candidates[0][2]

        if not approval_value or not eligibility_value:
            return ""

        return (
            "- **Approval:** " + approval_value + "\n"
            "- **Eligibility:** " + eligibility_value + "\n"
            "- **Scope:** Based on these two statements from the same accepted "
            "company document, the approval requirement applies to requests under "
            "the listed benefit for the stated eligible group. The source does not "
            "establish a broader population beyond what it explicitly states."
        )

    def _grounded_use_eligibility_answer(
        self,
        context: str,
        question: str,
    ) -> str:
        """Resolve natural "who can use it?" wording to explicit eligibility.

        This path activates only when the accepted source itself exposes an
        Eligibility/Qualification field.  It therefore does not reinterpret a
        generic authorization question unless the company document explicitly
        provides the entitlement relation needed to answer it.
        """

        if not context or not question:
            return ""

        clean_question = self._normalize_identity_match_text(question)
        if not re.search(
            r"\b(?:who\s+can\s+use|who\s+may\s+use|who\s+is\s+it\s+for|"
            r"sino\s+(?:ang\s+)?(?:pwede|puwede|pwedeng|puwedeng|maaaring)\s+(?:gumamit|makagamit))\b",
            clean_question,
        ):
            return ""

        explicit_eligibility = False
        for label, _value in self._compact_label_value_pairs_from_context(
            context,
            max_chars=5000,
        ):
            label_norm = self._normalize_identity_match_text(label)
            if re.search(
                r"\b(?:eligibility|eligible|qualification|qualified|entitlement)\b",
                label_norm,
            ):
                explicit_eligibility = True
                break

        if not explicit_eligibility:
            return ""

        return self._grounded_eligibility_answer(
            context=context,
            question=question,
            answer_focus=(
                "ELIGIBLE OR ENTITLED ENTITY: Resolve the explicitly stated "
                "eligible/entitled population from accepted company context."
            ),
        )

    def _grounded_scope_followup_answer(
        self,
        context: str,
        question: str,
    ) -> str:
        """Answer narrow deictic scope checks directly from accepted text.

        Example: after retrieving a rule saying "either directly or
        indirectly", the follow-up "Does it apply indirectly too?" can be
        answered without asking an LLM to rediscover the already explicit
        source phrase.  Only literal scope terms present in the accepted
        context qualify.
        """

        if not context or not question:
            return ""

        clean_question = re.sub(
            r"\s+",
            " ",
            str(question or "").strip(),
        )
        match = re.match(
            r"(?i)^does\s+(?:it|this|that)\s+apply\s+(?P<scope>.+?)"
            r"(?:\s+too)?\s*[?.!]*$",
            clean_question,
        )
        if not match:
            return ""

        scope = re.sub(r"\s+", " ", match.group("scope")).strip(" .?!")
        scope_tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", scope.casefold())
            if token not in {"too", "also", "the", "a", "an"}
        ]
        if not scope_tokens:
            return ""

        source_text = re.sub(
            r"(?m)^=====.*?=====\s*$",
            " ",
            str(context),
        )
        source_text = re.sub(r"\s+", " ", source_text).strip()
        candidates = re.split(r"(?<=[.!?])\s+|\s{2,}", source_text)
        for sentence in candidates:
            normalized = sentence.casefold()
            if all(re.search(rf"\b{re.escape(token)}\b", normalized) for token in scope_tokens):
                cleaned = re.sub(r"\s+", " ", sentence).strip()
                if cleaned:
                    return f"Yes. {cleaned}"

        return ""

    def _should_verify_answer_focus(
        self,
        answer_focus: str
    ):

        if not answer_focus:

            return False

        prefixes = (
            "IDENTITY OR OVERVIEW:",
            "SHORT TOPIC OVERVIEW:",
            "ELIGIBLE OR ENTITLED ENTITY:",
            "AUTHORIZED ENTITY:",
            "RESPONSIBLE ENTITY:",
            "APPROVER:",
            "PERSON OR ENTITY:",
            "ENTITY OR CHOICE:",
            "QUANTITY:",
            "TIME:",
            "LOCATION:",
            "REASON:",
            "PURPOSE:",
            "EFFECT:",
            "SUBJECT MATTER:",
            "COMPARISON:",
            "YES OR NO:",
        )

        return answer_focus.startswith(
            prefixes
        )

    def _strip_query_echo_from_answer(
        self,
        answer: str,
        question: str = "",
        resolved_question: str = "",
    ):

        """Remove a leading echoed user/resolved query without changing facts.

        Small local verifier models can prepend the resolved search target as
        a standalone line and then provide the actual answer. The echoed query
        is internal routing text, not answer content, so remove it only when
        the first non-empty line is an exact normalized match for the current
        question or resolved target.
        """

        if not answer:
            return answer

        lines = str(answer).splitlines()
        first_index = next(
            (index for index, line in enumerate(lines) if line.strip()),
            None,
        )

        if first_index is None:
            return answer

        first = lines[first_index].strip().strip('"\'` ')
        first_norm = self._normalize_identity_match_text(first)
        targets = {
            self._normalize_identity_match_text(value)
            for value in (question, resolved_question)
            if value
        }
        targets.discard("")

        if first_norm not in targets:
            return answer

        remaining = lines[:first_index] + lines[first_index + 1:]
        cleaned = "\n".join(remaining).strip()

        if cleaned:
            evidence_logger.record_event(
                event_name="OUTPUT CLEANUP",
                status="QUERY ECHO REMOVED",
                details={
                    "reason": (
                        "Verifier output echoed the current question/resolved "
                        "target before the actual grounded answer."
                    )
                },
            )
            return cleaned

        return answer

    def _importance_draft_can_skip_focus_verifier(
        self,
        draft_answer: str,
        question: str,
        resolved_question: str = "",
    ):

        """Skip a redundant verifier for clearly grounded importance answers.

        This is deliberately narrower than generic WHY handling. It applies
        only to importance/significance wording, where a concise factual role
        or impact statement can itself answer why the subject matters. Other
        causal/rationale questions keep the verifier path.
        """

        if not draft_answer:
            return False

        target = self._normalize_identity_match_text(
            resolved_question or question
        )
        if not re.search(
            r"\b(?:important|importance|significant|significance)\b",
            target,
        ):
            return False

        clean = self._remove_invalid_yes_no_prefix(
            draft_answer,
            "REASON:",
        )
        clean = re.sub(r"\s+", " ", clean).strip()

        if (
            not clean
            or clean.lower() == NO_RESULT_MESSAGE.strip().lower()
            or self._contains_prompt_leak(clean)
            or len(clean.split()) < 8
            or len(clean.split()) > 90
        ):
            return False

        ignored = {
            "why", "was", "is", "were", "are", "did", "does", "do",
            "important", "importance", "significant", "significance",
            "he", "him", "his", "she", "her", "hers", "they", "them",
            "their", "it", "its", "the", "a", "an", "of", "to", "for",
            "in", "on", "what", "who", "sino", "bakit", "siya", "ang",
        }
        subject_tokens = [
            token
            for token in target.split()
            if len(token) >= 2 and token not in ignored
        ][:6]
        answer_norm = self._normalize_identity_match_text(clean)

        if subject_tokens:
            required = min(2, len(subject_tokens))
            if sum(token in answer_norm for token in subject_tokens) < required:
                return False

        # Require a complete statement rather than a bare name/value.
        return bool(re.search(r"\b(?:is|was|are|were|served|became|led|helped|shaped|enabled|established|founded|created)\b", answer_norm))

    def _short_direct_draft_can_skip_focus_verifier(
        self,
        context: str,
        question: str,
        draft_answer: str,
        answer_focus: str,
        resolved_question: str = "",
    ):

        """Skip a verifier only when a short value is locally proven in context.

        The accepted retrieval context remains the source-of-truth.  A value
        must occur next to relation-specific evidence, so merely appearing
        somewhere else in the same chunk is not enough.
        """

        if not context or not draft_answer or not answer_focus:
            return False

        allowed_prefixes = (
            "APPROVER:",
            "QUANTITY:",
            "TIME:",
            "LOCATION:",
        )
        if not answer_focus.startswith(allowed_prefixes):
            return False

        clean = self._remove_invalid_yes_no_prefix(
            draft_answer,
            answer_focus,
        )
        clean = re.sub(r"\s+", " ", clean).strip(" .")

        if (
            not clean
            or clean.lower() == NO_RESULT_MESSAGE.strip().lower()
            or self._contains_prompt_leak(clean)
            or len(clean.split()) > 14
        ):
            return False

        normalized_context = self._normalize_identity_match_text(context)
        normalized_answer = self._normalize_identity_match_text(clean)
        normalized_target = self._normalize_identity_match_text(
            resolved_question or question
        )

        if not normalized_answer:
            return False

        # Prefer explicit compact label/value pairing when available. This is
        # stronger than simple proximity and prevents a neighboring value
        # (for example vacation leave) from being accepted for another label
        # (for example sick leave).
        compact_pairs = self._compact_label_value_pairs_from_context(context)
        if compact_pairs:
            target_words = {
                token
                for token in normalized_target.split()
                if len(token) >= 3
            }

            if answer_focus.startswith("APPROVER:"):
                for label, value in compact_pairs:
                    label_norm = self._normalize_identity_match_text(label)
                    value_norm = self._normalize_identity_match_text(value)
                    if (
                        normalized_answer in value_norm
                        and (
                            "approval" in label_norm
                            or "approver" in label_norm
                            or re.search(
                                r"\b(?:approve|approval|approver|authorize|authorization)\b",
                                value_norm,
                            )
                        )
                    ):
                        return True
                return False

            if answer_focus.startswith("QUANTITY:"):
                ignored_pair_tokens = {
                    "how", "many", "much", "provided", "provide",
                    "provides", "given", "give", "gives", "days", "day",
                }
                relation_words = {
                    token for token in target_words
                    if token not in ignored_pair_tokens
                }
                for label, value in compact_pairs:
                    label_norm = self._normalize_identity_match_text(label)
                    value_norm = self._normalize_identity_match_text(value)
                    label_words = set(label_norm.split())
                    if (
                        normalized_answer in value_norm
                        and relation_words
                        and len(relation_words.intersection(label_words))
                        >= min(2, len(relation_words))
                    ):
                        return True
                return False

        answer_match = re.search(
            rf"\b{re.escape(normalized_answer)}\b",
            normalized_context,
        )
        if not answer_match:
            return False

        start = max(0, answer_match.start() - 160)
        end = min(len(normalized_context), answer_match.end() + 160)
        local_window = normalized_context[start:end]

        if answer_focus.startswith("APPROVER:"):
            return bool(
                re.search(
                    r"\b(?:approve|approves|approved|approval|approver|"
                    r"authorize|authorizes|authorized|authorization)\b",
                    local_window,
                )
            )

        ignored = {
            "how", "many", "much", "when", "where", "what", "which",
            "the", "a", "an", "is", "are", "was", "were", "did",
            "does", "do", "of", "to", "for", "in", "on", "at", "and",
            "provided", "provide", "provides", "given", "give", "gives",
            "date", "time", "location", "place", "days", "day",
        }
        target_tokens = [
            token
            for token in normalized_target.split()
            if len(token) >= 3 and token not in ignored
        ]
        target_tokens = list(dict.fromkeys(target_tokens))[:8]

        if not target_tokens:
            return False

        required = 1 if len(target_tokens) == 1 else 2
        local_hits = sum(
            1
            for token in target_tokens
            if re.search(rf"\b{re.escape(token)}\b", local_window)
        )

        return local_hits >= min(required, len(target_tokens))

    @staticmethod
    def _context_relevance_tokens(text: str):
        normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            str(text or "").lower(),
        )
        ignored = {
            "the", "a", "an", "of", "to", "for", "in", "on", "at",
            "and", "or", "is", "are", "was", "were", "be", "been",
            "what", "who", "when", "where", "why", "how", "which",
            "does", "did", "do", "can", "may", "must", "should",
            "please", "tell", "explain", "describe", "give", "me",
        }
        return [
            token for token in normalized.split()
            if len(token) >= 3 and token not in ignored
        ]

    def _compact_one_fast_context_chunk(
        self,
        text: str,
        relevance_tokens,
        max_chars: int = 2200,
    ):

        """Keep the opening identity/title area plus the densest relation window."""

        clean = str(text or "").strip()
        if len(clean) <= max_chars:
            return clean

        head_chars = min(900, max_chars // 2)
        head = clean[:head_chars].rstrip()
        lowered = "".join(
            character
            for character in unicodedata.normalize("NFKD", clean.lower())
            if not unicodedata.combining(character)
        )

        positions = []
        for token in relevance_tokens:
            pattern = (
                rf"\b{re.escape(token)}"
                if len(token) >= 4
                else rf"\b{re.escape(token)}\b"
            )
            positions.extend(
                match.start()
                for match in re.finditer(pattern, lowered)
            )

        if not positions:
            return clean[:max_chars].rstrip()

        window_size = max_chars - head_chars - 12
        best_start = 0
        best_hits = -1

        for position in positions:
            start = max(0, position - window_size // 3)
            end = min(len(clean), start + window_size)
            hits = sum(start <= other < end for other in positions)
            if hits > best_hits:
                best_hits = hits
                best_start = start

        best_end = min(len(clean), best_start + window_size)

        if best_start <= head_chars + 80:
            return clean[:max_chars].rstrip()

        tail = clean[best_start:best_end].strip()
        return (head + "\n...\n" + tail)[:max_chars].rstrip()

    def _compact_fast_generation_context(
        self,
        context: str,
        results,
        question: str,
        resolved_question: str,
        answer_focus: str,
        max_total_chars: int = 4800,
    ):

        """Reduce prompt payload for simple fast-model questions only.

        Full accepted context remains available to every deterministic safety
        and verification step.  This compact copy is used solely for the first
        fast-model generation call.
        """

        if not context or len(context) <= max_total_chars:
            return context

        focus = str(answer_focus or "")
        if focus.startswith((
            "COMPOUND:",
            "LIST:",
            "GROUNDED EXPLANATION:",
            "STRUCTURED EXPLANATION:",
        )):
            return context

        if self._is_multi_answer_question(question):
            return context

        tokens = self._context_relevance_tokens(
            " ".join(
                value for value in (resolved_question, question)
                if value
            )
        )

        pieces = []
        remaining = max_total_chars

        for index, item in enumerate(list(results or [])[:3], start=1):
            if remaining <= 180:
                break

            text = str(item.get("text", "") or "").strip()
            if not text:
                continue

            header = f"===== DOCUMENT {index} =====\n"
            allowance = min(2200, remaining - len(header))
            if allowance <= 120:
                break

            compact = self._compact_one_fast_context_chunk(
                text=text,
                relevance_tokens=tokens,
                max_chars=allowance,
            )
            piece = header + compact
            pieces.append(piece)
            remaining -= len(piece) + 2

        compact_context = "\n\n".join(pieces).strip()

        if (
            not compact_context
            or len(compact_context) >= len(context)
        ):
            return context

        evidence_logger.record_event(
            event_name="LATENCY OPTIMIZATION",
            status="FAST CONTEXT COMPACTED",
            details={
                "original_chars": len(context),
                "generation_chars": len(compact_context),
                "reduction_percent": round(
                    100.0 * (len(context) - len(compact_context)) / len(context),
                    1,
                ),
                "note": (
                    "Only the initial fast-model prompt was compacted; full "
                    "accepted context remains available to safety/verifier steps."
                ),
            },
        )

        return compact_context

    def _verify_answer_focus(
        self,
        context: str,
        question: str,
        draft_answer: str,
        answer_focus: str,
        resolved_question: str = "",
        llm_client=None,
    ):

        """
        Correct a true but wrongly focused draft using only
        the retrieved company knowledge.
        """

        fallback = NO_RESULT_MESSAGE.strip()

        if (
            not context
            or not draft_answer
        ):

            return draft_answer

        if not self._should_verify_answer_focus(
            answer_focus
        ):

            return draft_answer

        grounded_relation = self._grounded_direct_relation_answer(
            context=context,
            question=question,
            answer_focus=answer_focus,
        )

        if grounded_relation:
            evidence_logger.record_event(
                event_name="GROUNDED RELATION SAFETY",
                status="DIRECT RELATION PRESERVED",
                details={
                    "reason": (
                        "Accepted company context contained an explicit lexical "
                        "relation for the requested cause, organization, "
                        "position, role, or title."
                    )
                },
            )
            return grounded_relation

        if (
            draft_answer.strip().lower()
            == fallback.lower()
        ):

            return draft_answer

        # Do not let an invalid standalone Yes/No prefix influence the verifier.
        # The final presentation already removes this prefix; doing it here also
        # makes direct-answer skip gates evaluate the factual sentence itself.
        draft_answer = self._remove_invalid_yes_no_prefix(
            draft_answer,
            answer_focus,
        )

        if self._short_direct_draft_can_skip_focus_verifier(
            context=context,
            question=question,
            draft_answer=draft_answer,
            answer_focus=answer_focus,
            resolved_question=resolved_question,
        ):
            evidence_logger.record_event(
                event_name="LATENCY OPTIMIZATION",
                status="FOCUS VERIFIER SKIPPED",
                details={
                    "reason": (
                        "Short direct value was explicitly supported in the "
                        "local relation window of accepted company context."
                    )
                },
            )
            return draft_answer

        if (
            answer_focus.startswith("REASON:")
            and self._importance_draft_can_skip_focus_verifier(
                draft_answer=draft_answer,
                question=question,
                resolved_question=resolved_question,
            )
        ):
            evidence_logger.record_event(
                event_name="LATENCY OPTIMIZATION",
                status="FOCUS VERIFIER SKIPPED",
                details={
                    "reason": (
                        "Grounded importance/significance draft already directly "
                        "answers the resolved subject relation."
                    )
                },
            )
            return draft_answer

        if (
            answer_focus.startswith("IDENTITY OR OVERVIEW:")
            and self._identity_draft_can_skip_focus_verifier(
                draft_answer=draft_answer,
                question=question,
                resolved_question=resolved_question,
            )
        ):
            evidence_logger.record_event(
                event_name="LATENCY OPTIMIZATION",
                status="FOCUS VERIFIER SKIPPED",
                details={
                    "reason": (
                        "Concise identity draft already visibly satisfies the "
                        "requested identity/overview shape."
                    )
                },
            )
            return draft_answer

        verify_prompt = f"""
You are validating whether a draft answer directly answers the user's question.

Use ONLY the COMPANY KNOWLEDGE below.
Do not use outside knowledge.
Do not guess.
Do not invent facts.
Do not mention documents, context, sources, prompts, or verification.

COMPANY KNOWLEDGE:
{context}

USER QUESTION:
{question}

RESOLVED QUESTION OR TARGET:
{resolved_question or question}

REQUIRED ANSWER FOCUS:
{answer_focus}

DRAFT ANSWER:
{draft_answer}

Rules:
1. Check whether the DRAFT ANSWER matches the REQUIRED ANSWER FOCUS.
2. Match the answer to the exact relation or target expressed by the
   RESOLVED QUESTION OR TARGET. If several values of the same type appear
   in COMPANY KNOWLEDGE, choose only the value explicitly connected to the
   requested relation or target.
3. A PERSON, ELIGIBLE, ENTITLED, AUTHORIZED, RESPONSIBLE, APPROVER,
   ENTITY, or CHOICE question must answer with the exact requested person,
   group, role, team, organization, system, component, category, item,
   option, or entity that satisfies the relationship in the question.
4. Do not answer an entity-type question with only a quantity, time,
   location, reason, definition, or procedure.
4. A HOW MANY or HOW MUCH question must answer with the quantity and unit.
5. A WHEN question must answer with the date, time, period, deadline,
   schedule, or triggering condition.
6. A WHERE question must answer with the location, section, path, system area,
   or place.
7. A WHY question must answer with the stated reason, rationale, purpose,
   or cause.
8. If the REQUIRED ANSWER FOCUS is IDENTITY OR OVERVIEW or SHORT TOPIC
   OVERVIEW, identify the subject and provide a brief useful description.
   Do not return only the name or title when COMPANY KNOWLEDGE contains an
   explicitly supported role, purpose, definition, significance, or key detail.
9. If the REQUIRED ANSWER FOCUS is APPROVER, return the approver,
   approving role, approval authority, or approving entity itself.
   Do not merely state that approval is required.
10. If the REQUIRED ANSWER FOCUS is an entity type, return the entity directly.
11. Do not start with Yes or No unless the REQUIRED ANSWER FOCUS is YES OR NO.
12. If the REQUIRED ANSWER FOCUS is STRUCTURED STATEMENT, return the exact
    requested Rule or Directive statement itself; do not replace it with its
    rationale, applicability, category, or a nearby cross-reference.
13. If the REQUIRED ANSWER FOCUS is STRUCTURED EXPLANATION, state the exact
    requested Rule or Directive first, then add a concise explanation or
    rationale explicitly tied to that same identifier when available.
14. If the draft already has the correct focus and facts, return it unchanged.
15. If the draft has the wrong focus, return a corrected direct answer using
    only explicitly stated COMPANY KNOWLEDGE.
16. Return a complete grammatical answer.
17. Correct singular/plural agreement when the factual meaning stays unchanged.
18. If the requested focused answer is not explicitly supported, return exactly:
{fallback}
19. Return only the final answer.
20. Never output labels or commentary such as DRAFT ANSWER,
    MATCHED ANSWER FOCUS, REQUIRED ANSWER FOCUS, VERIFICATION,
    CORRECTED ANSWER, or ANSWER:.
21. Your entire output must be either the final direct answer text or the
    exact fallback sentence from rule 18.
"""

        try:

            verifier = llm_client or self.llm
            verifier_output = self._generate_with_latency(
                verifier,
                verify_prompt,
                purpose="answer_focus_verification",
            )

            verified_answer = (
                self._extract_verifier_answer(
                    verifier_output,
                    draft_answer
                )
            )

            verified_answer = self._strip_query_echo_from_answer(
                verified_answer,
                question=question,
                resolved_question=resolved_question,
            )

            if not verified_answer:

                return draft_answer

            verified_answer = self._preserve_grounded_identity_overview(
                context=context,
                question=question,
                resolved_question=resolved_question,
                answer_focus=answer_focus,
                verified_answer=verified_answer
            )

            if DEBUG_MODE:

                print(
                    "\n===== ANSWER FOCUS VERIFICATION ====="
                )

                print(
                    f"Required focus : "
                    f"{answer_focus}"
                )

                print(
                    f"Draft answer   : "
                    f"{draft_answer}"
                )

                print(
                    f"Verified answer: "
                    f"{verified_answer}"
                )

                print(
                    "=====================================\n"
                )

            return verified_answer

        except Exception:

            return draft_answer

    def _extract_verifier_answer(
        self,
        verifier_output: str,
        draft_answer: str = ""
    ):

        """Extract only the answer text from an over-verbose verifier.

        Small local models sometimes ignore the verifier instruction and
        return a transcript such as ``DRAFT ANSWER: ...`` followed by
        ``MATCHED ANSWER FOCUS`` and ``ANSWER: ...``. That internal
        commentary must never reach the safety gate or UI.
        """

        if not verifier_output:
            return draft_answer

        fallback = NO_RESULT_MESSAGE.strip()
        raw = verifier_output.strip()

        if raw.lower() == fallback.lower():
            return fallback

        # Prefer the last explicit final-answer marker because verifier
        # transcripts often repeat the draft before the corrected answer.
        answer_matches = list(
            re.finditer(
                r"(?im)^\s*(?:final\s+answer|corrected\s+answer|answer)\s*:\s*",
                raw
            )
        )

        if answer_matches:

            candidate = raw[
                answer_matches[-1].end():
            ].strip()

            # Stop if another known verifier metadata label somehow follows.
            candidate = re.split(
                r"(?im)^\s*(?:draft\s+answer|matched\s+answer\s+focus|"
                r"required\s+answer\s+focus|verification)\s*:\s*",
                candidate,
                maxsplit=1
            )[0].strip()

            candidate = self._postprocess_answer(
                candidate
            )

            if candidate:
                return candidate

        verifier_labels = (
            r"(?im)^\s*draft\s+answer\s*:",
            r"(?im)^\s*matched\s+answer\s+focus\s*:",
            r"(?im)^\s*required\s+answer\s+focus\s*:",
            r"(?im)^\s*verification\s*:",
        )

        if any(
            re.search(
                pattern,
                raw
            )
            for pattern in verifier_labels
        ):

            # Metadata-only verifier output is not a trustworthy corrected
            # answer. Preserve the already generated draft rather than
            # passing internal labels into the output safety gate.
            return draft_answer

        return self._postprocess_answer(
            raw
        )

    def _remove_invalid_yes_no_prefix(
        self,
        answer: str,
        answer_focus: str
    ):

        """
        Remove a standalone Yes/No prefix when the question is
        not a yes-or-no question.

        This changes only presentation, not factual content.
        """

        if not answer:

            return answer

        if (
            answer_focus
            and answer_focus.startswith(
                "YES OR NO:"
            )
        ):

            return answer

        lines = answer.splitlines()

        while (
            lines
            and lines[0].strip().lower()
            in {
                "yes",
                "yes.",
                "no",
                "no."
            }
        ):

            lines.pop(0)

        cleaned = "\n".join(
            lines
        ).strip()

        return cleaned or answer

    def _temporal_relation_terms(
        self,
        question: str,
        resolved_question: str = ""
    ):

        """
        Build generic relation terms for date/time questions.

        The groups describe common temporal relations across
        policies, manuals, people, systems, releases, and events.
        They are not tied to a document, person, or test file.
        """

        source_text = (
            f"{question or ''} "
            f"{resolved_question or ''}"
        ).lower()

        relation_groups = [
            {
                "birth",
                "birthday",
                "born",
                "date of birth",
            },
            {
                "death",
                "died",
                "die",
                "deceased",
                "execution",
                "executed",
                "date of death",
            },
            {
                "publish",
                "published",
                "publication",
                "release",
                "released",
                "issued",
            },
            {
                "start",
                "started",
                "begin",
                "began",
                "commence",
                "commenced",
                "effective",
                "take effect",
                "takes effect",
            },
            {
                "end",
                "ended",
                "finish",
                "finished",
                "expire",
                "expired",
                "expiration",
            },
            {
                "create",
                "created",
                "creation",
                "found",
                "founded",
                "establish",
                "established",
            },
            {
                "approve",
                "approved",
                "approval",
                "authorize",
                "authorized",
            },
            {
                "install",
                "installed",
                "installation",
                "deploy",
                "deployed",
                "deployment",
            },
            {
                "update",
                "updated",
                "modify",
                "modified",
                "revision",
                "revised",
            },
        ]

        active_terms = set()
        opposing_terms = set()

        for group in relation_groups:

            if any(
                term in source_text
                for term in group
            ):

                active_terms.update(
                    group
                )

        # Birth and death are common opposing date relations.
        birth_group = relation_groups[0]
        death_group = relation_groups[1]

        if active_terms.intersection(
            birth_group
        ):

            opposing_terms.update(
                death_group
            )

        if active_terms.intersection(
            death_group
        ):

            opposing_terms.update(
                birth_group
            )

        return (
            active_terms,
            opposing_terms
        )

    def _relation_aware_date_answer(
        self,
        context: str,
        question: str,
        resolved_question: str,
        answer_focus: str,
        current_answer: str
    ):

        """
        Correct a date only when one unique date is directly
        connected to the requested temporal relation.

        This is intentionally conservative:
        - direct phrases such as "started on DATE" are accepted;
        - broad proximity across a paragraph or table is not enough;
        - when evidence is ambiguous, the existing answer is kept.

        The rule is generic and is not tied to a person,
        document, policy, or testing topic.
        """

        if (
            not context
            or not answer_focus
            or not answer_focus.startswith(
                "TIME:"
            )
        ):

            return current_answer

        active_terms, opposing_terms = (
            self._temporal_relation_terms(
                question,
                resolved_question
            )
        )

        if not active_terms:

            return current_answer

        date_pattern = re.compile(
            r"\b(?:"
            r"(?:January|February|March|April|May|June|"
            r"July|August|September|October|November|December)"
            r"\s+\d{1,2}(?:st|nd|rd|th)?"
            r"(?:,\s*|\s+)\d{4}"
            r"|"
            r"\d{1,2}(?:st|nd|rd|th)?\s+"
            r"(?:January|February|March|April|May|June|"
            r"July|August|September|October|November|December)"
            r"\s+\d{4}"
            r"|"
            r"\d{4}-\d{2}-\d{2}"
            r")\b",
            flags=re.IGNORECASE
        )

        active_pattern = (
            r"(?:"
            + "|".join(
                sorted(
                    (
                        re.escape(term)
                        for term in active_terms
                    ),
                    key=len,
                    reverse=True
                )
            )
            + r")"
        )

        opposing_pattern = (
            r"(?:"
            + "|".join(
                sorted(
                    (
                        re.escape(term)
                        for term in opposing_terms
                    ),
                    key=len,
                    reverse=True
                )
            )
            + r")"
            if opposing_terms
            else ""
        )

        direct_dates = {}

        for match in date_pattern.finditer(
            context
        ):

            date_value = match.group(0).strip()

            before = context[
                max(
                    0,
                    match.start() - 90
                ):
                match.start()
            ].lower()

            after = context[
                match.end():
                min(
                    len(context),
                    match.end() + 90
                )
            ].lower()

            # A direct relation must be close to the date.
            before_match = re.search(
                active_pattern
                + r"[^.\n;:]{0,45}$",
                before,
                flags=re.IGNORECASE
            )

            after_match = re.match(
                r"^[^.\n;:]{0,45}"
                + active_pattern,
                after,
                flags=re.IGNORECASE
            )

            is_direct = bool(
                before_match
                or after_match
            )

            if not is_direct:

                continue

            # Reject a candidate when an opposing temporal
            # relation is even closer to the same date.
            if opposing_pattern:

                opposing_before = re.search(
                    opposing_pattern
                    + r"[^.\n;:]{0,45}$",
                    before,
                    flags=re.IGNORECASE
                )

                opposing_after = re.match(
                    r"^[^.\n;:]{0,45}"
                    + opposing_pattern,
                    after,
                    flags=re.IGNORECASE
                )

                if (
                    opposing_before
                    or opposing_after
                ):

                    continue

            direct_dates[
                date_value.lower()
            ] = date_value

        # Override only when the context provides exactly one
        # unambiguous direct relation-to-date match.
        if len(direct_dates) != 1:

            return current_answer

        selected_date = next(
            iter(
                direct_dates.values()
            )
        )

        current_dates = {
            match.group(0).strip().lower()
            for match in date_pattern.finditer(
                current_answer or ""
            )
        }

        # Keep the current answer when it already contains
        # the unique directly supported date.
        if (
            selected_date.lower()
            in current_dates
        ):

            return current_answer

        if DEBUG_MODE:

            print(
                "\n===== RELATION-AWARE DATE CHECK ====="
            )

            print(
                f"Resolved target : "
                f"{resolved_question or question}"
            )

            print(
                f"Current answer  : "
                f"{current_answer}"
            )

            print(
                f"Selected date   : "
                f"{selected_date}"
            )

            print(
                "Evidence type    : "
                "unique direct relation match"
            )

            print(
                "=====================================\n"
            )

        return selected_date


    def _build_prompt(
        self,
        context,
        question,
        history,
        answer_focus
    ):

        # Insert context and question into prompt template
        user_prompt = (
            ANSWER_TEMPLATE.format(
                context=context,
                history=history,
                question=question,
                answer_focus=answer_focus
            )
        )

        # Combine system instructions and user prompt
        prompt = (
            SYSTEM_PROMPT
            + "\n\n"
            + user_prompt
        )

        return prompt

    def _extract_sources(
        self,
        results,
        max_sources: int | None = 1,
    ):

        sources = []

        for item in results:

            metadata = item["metadata"]

            reference = metadata.get("exact_reference")
            if not reference:
                section_type = str(metadata.get("section_type", "") or "").strip().casefold()
                rule_id = str(metadata.get("rule_id", "") or "").strip()
                section_id = str(metadata.get("section_id", "") or "").strip()
                if rule_id and section_type == "rule":
                    reference = f"Rule {rule_id}"
                elif rule_id and section_type == "directive":
                    reference = f"Dir {rule_id}"
                elif section_id and section_type == "section":
                    reference = f"Section {section_id}"
                else:
                    reference = (
                        metadata.get("directive_id")
                        or metadata.get("section_title")
                        or metadata.get("section")
                    )

            source = {
                "name": decode_unicode_markers(metadata.get("file_name", "Unknown")),
                "path": metadata.get("file_path", ""),
                "page_start": metadata.get("page_start") or metadata.get("page_number") or metadata.get("page"),
                "page_end": metadata.get("page_end") or metadata.get("page_start") or metadata.get("page_number") or metadata.get("page"),
                "reference": reference,
            }

            if source not in sources:

                sources.append(source)

        # Preserve the long-standing single-source display for ordinary
        # answers. Deterministic cross-source comparisons may explicitly ask
        # for all source records that are required to support both sides.
        if max_sources is None:
            return sources

        safe_limit = max(0, int(max_sources))
        return sources[:safe_limit]

    @staticmethod
    def _temporal_comparison_source_results(results, answer: str):
        """Return the minimum ranked result set that grounds each answer date.

        Cross-document temporal comparisons are only finalized
        deterministically when both requested event dates are present in the
        accepted context.  The normal UI intentionally shows only the highest
        ranked source, but that would hide one side of a cross-document
        comparison.  This helper keeps one accepted result for every distinct
        full calendar date used by the deterministic answer.
        """

        if not results or not answer:
            return []

        month_names = (
            r"January|February|March|April|May|June|July|August|"
            r"September|October|November|December"
        )
        date_pattern = re.compile(
            rf"\b(?:"
            rf"(?:{month_names})\s+\d{{1,2}},?\s+(?:18|19|20)\d{{2}}"
            rf"|"
            rf"\d{{1,2}}\s+(?:{month_names})\s+(?:18|19|20)\d{{2}}"
            rf")\b",
            flags=re.IGNORECASE,
        )

        required_dates = []
        for match in date_pattern.finditer(str(answer)):
            label = re.sub(r"\s+", " ", match.group(0)).strip()
            if label.casefold() not in {item.casefold() for item in required_dates}:
                required_dates.append(label)

        if len(required_dates) < 2:
            return []

        selected = []
        selected_source_keys = set()

        for date_label in required_dates:
            for item in results:
                text = str(item.get("text", "") or "")
                if not re.search(
                    rf"\b{re.escape(date_label)}\b",
                    text,
                    flags=re.IGNORECASE,
                ):
                    continue

                metadata = item.get("metadata", {}) or {}
                source_key = (
                    metadata.get("file_path")
                    or metadata.get("file_name")
                    or id(item)
                )
                if source_key not in selected_source_keys:
                    selected.append(item)
                    selected_source_keys.add(source_key)
                break

        return selected if len(selected) >= 2 else []

    def _strip_output_wrappers(
        self,
        answer: str
    ):

        """
        Remove harmless model wrappers without changing answer facts.
        """

        if not answer:

            return answer

        answer = re.sub(
            r"(?im)^\s*(final\s+answer|answer|response)\s*:\s*",
            "",
            answer,
            count=1
        )

        answer = re.sub(
            r"(?im)^\s*(?:structured\s+statement|structured\s+detail|structured\s+explanation|"
            r"grounded\s+explanation|definition\s+or\s+detail|identity\s+or\s+overview|"
            r"short\s+topic\s+overview|reason|procedure|list|time|location|"
            r"quantity|entity\s+or\s+choice|person\s+or\s+entity|"
            r"authorized\s+entity|responsible\s+entity|approver|"
            r"eligible\s+or\s+entitled\s+entity|compound|general)\s*:\s*",
            "",
            answer,
            count=1
        )

        # Remove only a whole-answer markdown/text wrapper.  Internal fenced
        # code blocks are user-visible source formatting and must survive.
        lines = answer.splitlines()
        first_nonempty = next(
            (index for index, line in enumerate(lines) if line.strip()),
            None,
        )
        last_nonempty = next(
            (index for index in range(len(lines) - 1, -1, -1) if lines[index].strip()),
            None,
        )

        if (
            first_nonempty is not None
            and last_nonempty is not None
            and first_nonempty < last_nonempty
            and re.fullmatch(r"\s*```(?:markdown|text)?\s*", lines[first_nonempty], re.IGNORECASE)
            and re.fullmatch(r"\s*```\s*", lines[last_nonempty])
        ):
            del lines[last_nonempty]
            del lines[first_nonempty]
            answer = "\n".join(lines)

        return answer.strip()

    @staticmethod
    def _structured_detail_label(question: str, resolved_question: str = ""):
        """Return the canonical explicitly requested structured detail label."""

        text = re.sub(
            r"\s+",
            " ",
            f"{question or ''} {resolved_question or ''}".lower(),
        ).strip()

        if re.search(
            r"\b(?:category|classification|classified|mandatory|required|advisory)\b",
            text,
        ):
            return "Category"
        if re.search(
            r"\b(?:apply\s+to|applies\s+to|applicable\s+to|applicability|c\s+versions?|c\s+standards?)\b",
            text,
        ):
            return "Applies to"
        if re.search(r"\banalysis\b", text):
            return "Analysis"
        if re.search(r"\bamplification\b", text):
            return "Amplification"
        if re.search(r"\brationale\b", text):
            return "Rationale"
        if re.search(r"\bexamples?\b", text):
            return "Examples"
        if re.search(r"\bexceptions?\b", text):
            return "Exceptions"
        if re.search(r"\bsee\s+also\b", text):
            return "See also"
        return ""

    @staticmethod
    def _is_explicit_temporal_comparison_question(question: str):
        """Return True only for clear two-event before/after comparison forms.

        Keep this detector deliberately narrow so ordinary date questions do not
        get routed through comparison logic.  English and Tagalog variants share
        the same downstream two-event grounding requirements.
        """

        clean = re.sub(r"\s+", " ", str(question or "")).strip(" .?!")
        if not clean:
            return False

        patterns = (
            r"\b(?:before|after|earlier|later)\b.+\bthan\b",
            r"\b(?:before\s+or\s+after)\b",
            r"^\s*(?:was|is|did)\s+.+\s+(?:before|after|earlier\s+than|later\s+than)\s+.+$",
            r"\b(?:nauna|sumunod)\b.+\bkaysa\b",
            r"\bmas\s+nauna\b.+\bkaysa\b",
            r"^\s*which\s+came\s+(?:later|first|earlier)\s*:",
            r"^\s*alin\s+ang\s+(?:nauna|sumunod)\s*:",
            r"^\s*compare\s+.+\s+(?:with|to|and)\s+.+\b(?:date|dates|first|earlier|later|chronological)\b",
            r"\bchronological\s+order\b.*?:\s*.+\s+(?:and|or|at|o)\s+.+",
        )

        return any(
            re.search(pattern, clean, flags=re.IGNORECASE)
            for pattern in patterns
        )

    @staticmethod
    def _structured_missing_detail_message(label: str, display_name: str):
        """Explain a verified missing structured field without implying the document was absent."""

        singular = {
            "Examples": "Example",
            "Exceptions": "Exception",
        }.get(label, label)
        field_labels = {"Category", "Analysis", "Applies to"}
        noun = "field" if singular in field_labels else "section"
        return (
            f"No {singular} {noun} is provided for {display_name} "
            "in the available company knowledge."
        )

    def _structured_statement_from_context(
        self,
        context: str,
        reference
    ):

        """Extract the first direct statement for an exact Rule/Directive.

        Exact structured retrieval already returns the requested block first.
        This deterministic extraction prevents a small local model from
        answering with the rationale instead of the rule statement.
        """

        if (
            not context
            or reference is None
            or reference.kind not in {"rule", "directive"}
        ):
            return ""

        lines = [
            line.strip()
            for line in context.splitlines()
        ]

        if reference.kind == "rule":
            heading_re = re.compile(
                rf"^Rule\s+{re.escape(reference.identifier)}(?:\s*[:\-–—].*)?$",
                re.IGNORECASE
            )
        else:
            heading_re = re.compile(
                rf"^(?:Dir|Directive)\s+{re.escape(reference.identifier)}(?:\s*[:\-–—].*)?$",
                re.IGNORECASE
            )

        start_index = None

        for index, line in enumerate(lines):
            if heading_re.match(line):
                start_index = index
                break

        if start_index is None:
            return ""

        statement_lines = []
        stop_labels = re.compile(
            r"^(?:Category|Analysis|Applies\s+to|Rationale|Amplification|"
            r"Example|Examples|Exception|Exceptions|See\s+also|Notes?)\b",
            re.IGNORECASE
        )

        for line in lines[start_index + 1:]:
            if not line:
                if statement_lines:
                    break
                continue

            if line.startswith("=====") or stop_labels.match(line):
                break

            statement_lines.append(line)

        return re.sub(
            r"\s+",
            " ",
            " ".join(statement_lines)
        ).strip()

    def _structured_rationale_from_context(
        self,
        context: str
    ):

        """Return the first complete rationale sentence when present."""

        if not context:
            return ""

        match = re.search(
            r"(?ims)^\s*Rationale\s*$\s*(.+?)(?=^\s*(?:Amplification|Example|Examples|"
            r"Exception|Exceptions|See\s+also|Category|Analysis|Applies\s+to|Rule\s+\d|"
            r"Dir(?:ective)?\s+\d|(?:Section|Article|Chapter|Part)\s+[A-Za-z0-9IVXLCDM]+|={3,})\b|\Z)",
            context
        )

        if not match:
            return ""

        rationale = self._first_meaningful_rationale_sentence(
            match.group(1)
        )
        return rationale

    def _structured_labeled_block_from_context(
        self,
        context: str,
        label: str,
        preserve_layout: bool = False,
        stop_on_section_heading: bool = True,
    ):

        """Return one named subsection from an exact structured block.

        This keeps explanation enrichment deterministic and source-grounded.
        It never searches outside the already selected exact structured context.
        """

        if not context or not label:
            return ""

        label_heading = rf"^\s*{re.escape(label)}\s*:?\s*$"
        if label.casefold() == "see also":
            match = re.search(
                rf"(?ims){label_heading}\s*(.+?)(?=^\s*={{3,}}\s*$|\Z)",
                context,
            )
        else:
            named_boundary = (
                r"(?:Category|Analysis|Applies\s+to|Rationale|Amplification|"
                r"Example|Examples|Exception|Exceptions|See\s+also|Notes?)"
            )
            structured_boundary = r"(?:Rule\s+\d+(?:\.\d+)*|Dir(?:ective)?\s+\d+(?:\.\d+)*)"
            if stop_on_section_heading:
                structured_boundary += (
                    r"|(?:Section|Article|Chapter|Part)\s+"
                    r"[A-Za-z0-9IVXLCDM]+(?:\.\d+)*"
                )

            match = re.search(
                rf"(?ims){label_heading}\s*(.+?)(?="
                rf"^\s*{named_boundary}\s*:?\s*$|"
                rf"^\s*(?:{structured_boundary})\b|"
                rf"^\s*={{3,}}\s*$|\Z)",
                context
            )

        if not match:
            return ""

        if preserve_layout:
            raw = str(match.group(1) or "").replace("\r\n", "\n").replace("\r", "\n")
            lines = [line.rstrip() for line in raw.splitlines()]

            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()

            # Preserve source line order and indentation.  Limit only repeated
            # blank-line noise introduced by PDF extraction.
            cleaned_lines = []
            previous_blank = False
            for line in lines:
                is_blank = not line.strip()
                if is_blank and previous_blank:
                    continue
                cleaned_lines.append(line)
                previous_blank = is_blank

            return "\n".join(cleaned_lines).strip()

        return re.sub(
            r"\s+",
            " ",
            match.group(1)
        ).strip()

    @staticmethod
    def _structured_example_code_line(line: str):
        """Conservatively classify one extracted Example line as C source code.

        Natural-language MISRA examples frequently contain words such as
        ``for``, ``if``, ``struct`` or ``union`` in ordinary prose. Treating a
        keyword alone as code can fence prose incorrectly. Require concrete C
        syntax markers instead while still recognizing declarations,
        preprocessor lines, comments, control statements and function calls.
        """

        stripped = str(line or "").strip()
        if not stripped:
            return False

        if re.match(r"^(?:#\s*\w+|//|/\*|\*|\*/|\{|\}|[A-Za-z_]\w*:\s*(?:/\*.*\*/)?$)", stripped):
            return True

        if re.match(r"^(?:if|for|while|switch)\s*\(", stripped, flags=re.IGNORECASE):
            return True
        if re.match(r"^(?:else|do)\b(?:\s*\{|\s*)$", stripped, flags=re.IGNORECASE):
            return True
        if re.match(r"^(?:return|goto|break|continue)\b.*;\s*(?:/\*.*\*/)?$", stripped, flags=re.IGNORECASE):
            return True

        if re.match(
            r"^(?:typedef|extern|static|const|volatile|signed|unsigned|short|long|"
            r"void|char|int|float|double|struct|union|enum|bool|_Bool|"
            r"int\d+_t|uint\d+_t)\b",
            stripped,
            flags=re.IGNORECASE,
        ) and re.search(r"[;=(){}\[\]:]", stripped):
            return True

        if re.match(
            r"^[A-Za-z_]\w*(?:\s*\[[^\]]*\])?\s*"
            r"(?:=|\+=|-=|\*=|/=|%=|<<=|>>=|&=|\|=|\^=|\+\+|--)\s*.*;",
            stripped,
        ):
            return True
        if re.match(r"^(?:\+\+|--)?[A-Za-z_]\w*\s*(?:\+\+|--)?\s*;?$", stripped):
            return True
        if re.match(r"^[A-Za-z_]\w*\s*\([^)]*\)\s*;\s*(?:/\*.*\*/)?$", stripped):
            return True

        if re.search(r";\s*(?:/\*.*\*/)?$", stripped) and re.search(r"[=()\[\]+*/%<>!&|^-]", stripped):
            return True

        return False

    def _format_structured_example_answer(self, source_block: str, display_label: str = "Example"):
        """Render an explicit Example block without flattening code into prose.

        Source order is preserved.  PDF-wrapped prose is reflowed into readable
        paragraphs, while code-like runs keep their original extracted line
        breaks and indentation inside fenced code blocks.  No explanatory facts
        are generated or inferred.
        """

        raw = str(source_block or "").strip()
        if not raw:
            return ""

        lines = [
            line for line in raw.splitlines()
            if not re.match(
                r"^\s*Section\s+\d+\s*:\s*(?:Directives|Rules)\s*$",
                line,
                re.IGNORECASE,
            )
        ]
        flags = [self._structured_example_code_line(line) for line in lines]

        # Blank lines inside a code run belong to that run when code occurs on
        # both sides.  This preserves visually meaningful source spacing.
        for index, line in enumerate(lines):
            if line.strip():
                continue
            before = next((flags[pos] for pos in range(index - 1, -1, -1) if lines[pos].strip()), False)
            after = next((flags[pos] for pos in range(index + 1, len(lines)) if lines[pos].strip()), False)
            flags[index] = before and after

        segments = []
        current_kind = None
        current_lines = []

        def flush():
            nonlocal current_kind, current_lines
            if current_kind is None:
                return

            if current_kind == "code":
                value = "\n".join(current_lines).strip("\n")
            else:
                paragraphs = []
                paragraph = []
                for item in current_lines:
                    stripped_item = item.strip()
                    if not stripped_item:
                        if paragraph:
                            paragraphs.append(" ".join(paragraph).strip())
                            paragraph = []
                        continue

                    # Keep explicit source list/number markers on separate lines.
                    if re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", stripped_item):
                        if paragraph:
                            paragraphs.append(" ".join(paragraph).strip())
                            paragraph = []
                        paragraphs.append(stripped_item)
                    else:
                        paragraph.append(stripped_item)

                if paragraph:
                    paragraphs.append(" ".join(paragraph).strip())
                value = "\n\n".join(part for part in paragraphs if part)

            if value:
                segments.append((current_kind, value))

            current_kind = None
            current_lines = []

        for line, is_code in zip(lines, flags):
            kind = "code" if is_code else "prose"
            if current_kind is None:
                current_kind = kind
            elif kind != current_kind:
                flush()
                current_kind = kind
            current_lines.append(line)
        flush()

        if not segments:
            return ""

        label = "Examples" if str(display_label).strip().casefold() == "examples" else "Example"
        rendered = [f"{label}:"]
        for kind, value in segments:
            if kind == "code":
                rendered.append(f"```c\n{value}\n```")
            else:
                rendered.append(value)

        return "\n\n".join(rendered).strip()

    def _limit_grounded_explanation_text(
        self,
        text: str,
        max_sentences: int = 3,
        max_chars: int = 900
    ):

        """Keep a supported explanation useful without dumping a whole section."""

        clean = re.sub(
            r"\s+",
            " ",
            (text or "")
        ).strip()

        if not clean:
            return ""

        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9])",
            clean
        )

        selected = " ".join(
            sentence.strip()
            for sentence in sentences[:max_sentences]
            if sentence.strip()
        ).strip()

        if not selected:
            selected = clean

        if len(selected) <= max_chars:
            return selected

        shortened = selected[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")

        # A trailing ellipsis marks presentation truncation only; all returned
        # words still come directly from the exact structured context.
        return shortened + "..."

    def _format_structured_prose_block(self, source_block: str) -> str:
        """Reflow source prose while preserving list hierarchy and ordering.

        The structure-aware PDF extractor may wrap ordinary prose across visual
        lines.  Reflow those wraps for readability, but keep explicit bullets
        and numbered items as separate Markdown lines.  No factual text is
        generated or inferred.
        """

        raw = str(source_block or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not raw:
            return ""

        lines = [
            line for line in raw.splitlines()
            if not re.match(
                r"^\s*Section\s+\d+\s*:\s*(?:Directives|Rules)\s*$",
                line,
                re.IGNORECASE,
            )
        ]
        marker_re = re.compile(
            r"^\s*(?P<marker>[-*•]|\d+[.)]|[A-Za-z][.)])(?:\s+(?P<body>.*))?$"
        )
        blocks = []
        paragraph = []
        list_marker = ""
        list_parts = []

        def flush_paragraph():
            nonlocal paragraph
            if paragraph:
                blocks.append(" ".join(part.strip() for part in paragraph if part.strip()).strip())
                paragraph = []

        def flush_list_item():
            nonlocal list_marker, list_parts
            if list_marker:
                body = " ".join(part.strip() for part in list_parts if part.strip()).strip()
                blocks.append((list_marker + (" " + body if body else "")).rstrip())
            list_marker = ""
            list_parts = []

        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped:
                flush_list_item()
                flush_paragraph()
                if blocks and blocks[-1] != "":
                    blocks.append("")
                continue

            match = marker_re.match(raw_line)
            if match:
                flush_list_item()
                flush_paragraph()
                list_marker = match.group("marker")
                body = str(match.group("body") or "").strip()
                list_parts = [body] if body else []
                continue

            if list_marker:
                list_parts.append(stripped)
            else:
                paragraph.append(stripped)

        flush_list_item()
        flush_paragraph()

        cleaned = []
        previous_blank = False
        for block in blocks:
            is_blank = block == ""
            if is_blank and previous_blank:
                continue
            cleaned.append(block)
            previous_blank = is_blank

        return "\n".join(cleaned).strip()

    def _structured_explanation_from_context(
        self,
        context: str,
        reference,
        display_name: str
    ):

        """Explain an exact Rule/Directive without dumping unrequested metadata.

        Default explanation policy:
        - requirement statement
        - one concise rationale sentence when available
        - nothing else unless the user's intent explicitly requests it

        Category, Analysis, Applies-to mappings, amplification tables, examples,
        exceptions, notes and cross-references remain available through their
        dedicated structured intents.
        """

        statement = self._structured_statement_from_context(context, reference)
        statement = self._clean_structured_statement_for_display(statement)
        if not statement:
            return ""

        rationale = self._structured_rationale_from_context(context)
        rationale = self._first_meaningful_rationale_sentence(rationale)

        # Some Rules have no Rationale block but do have an Amplification.
        # In that case, use only its first meaningful sentence as explanation.
        if not rationale:
            amplification = self._structured_labeled_block_from_context(
                context,
                "Amplification",
                stop_on_section_heading=False,
            )
            rationale = self._first_meaningful_rationale_sentence(amplification)

        pieces = [
            f"**{display_name}**",
            statement.rstrip(".") + ".",
        ]
        if rationale:
            pieces.append(f"**Rationale:** {rationale}")
        return "\n\n".join(pieces).strip()

    def _section_heading_and_intro_from_context(
        self,
        context: str,
        reference
    ):

        """Return the exact Section title and its opening explanation."""

        if (
            not context
            or reference is None
            or not reference.is_section_like
        ):
            return "", ""

        lines = [
            line.strip()
            for line in context.splitlines()
            if line.strip()
            and not line.strip().startswith("=====")
        ]

        identifier = re.escape(reference.identifier)
        heading_re = re.compile(
            rf"^(?:(?:Section|Article|Chapter|Part)\s+)?{identifier}\s*(?::|[-–—])?\s*(.*)$",
            re.IGNORECASE
        )
        subsection_re = re.compile(
            rf"^(?:(?:Section|Article|Chapter|Part)\s+)?{identifier}\.\d+(?:\s+.*)?$",
            re.IGNORECASE
        )

        start_index = None
        title = ""

        for index, line in enumerate(lines):
            match = heading_re.match(line)
            if not match:
                continue

            # Do not confuse 6.1 / 6.2 with the Section 6 heading.
            if subsection_re.match(line):
                continue

            start_index = index
            title = match.group(1).strip(" :-–—")
            break

        if start_index is None:
            return "", ""

        intro_lines = []

        for line in lines[start_index + 1:]:
            if subsection_re.match(line):
                break

            repeated = heading_re.match(line)
            if repeated and not subsection_re.match(line):
                break

            intro_lines.append(line)

        intro = self._limit_grounded_explanation_text(
            " ".join(intro_lines),
            max_sentences=2,
            max_chars=650
        )

        return title, intro

    def _section_subtopic_titles_from_context(
        self,
        context: str,
        reference,
        max_topics: int = 6
    ):

        """Extract direct child subsection titles from one exact Section."""

        if (
            not context
            or reference is None
            or not reference.is_section_like
        ):
            return []

        lines = [
            line.strip()
            for line in context.splitlines()
            if line.strip()
            and not line.strip().startswith("=====")
        ]

        identifier = re.escape(reference.identifier)
        child_re = re.compile(
            rf"^(?:(?:Section|Article|Chapter|Part)\s+)?{identifier}\.(\d+)(?:\s+(.*))?$",
            re.IGNORECASE
        )

        topics = []
        seen = set()

        for index, line in enumerate(lines):
            match = child_re.match(line)
            if not match:
                continue

            title = (match.group(2) or "").strip(" :-–—")

            if not title and index + 1 < len(lines):
                candidate = lines[index + 1].strip()
                if (
                    candidate
                    and len(candidate) <= 100
                    and not re.match(r"^\d+(?:\.\d+)+\b", candidate)
                    and not re.match(
                        rf"^Section\s+{identifier}\b",
                        candidate,
                        re.IGNORECASE
                    )
                    and not candidate.endswith((".", "!", "?"))
                ):
                    title = candidate

            normalized = re.sub(r"\s+", " ", title).strip().casefold()

            if not normalized or normalized in seen:
                continue

            seen.add(normalized)
            topics.append(re.sub(r"\s+", " ", title).strip())

            if len(topics) >= max_topics:
                break

        return topics

    def _section_subtopic_summaries_from_context(
        self,
        context: str,
        reference,
        max_topics: int = 4
    ):

        """Return direct child subsection titles with one grounded key sentence."""

        if (
            not context
            or reference is None
            or not reference.is_section_like
        ):
            return []

        lines = [
            line.strip()
            for line in context.splitlines()
            if line.strip()
            and not line.strip().startswith("=====")
        ]

        identifier = re.escape(reference.identifier)
        child_re = re.compile(
            rf"^(?:(?:Section|Article|Chapter|Part)\s+)?{identifier}\.(\d+)(?:\s+(.*))?$",
            re.IGNORECASE
        )
        repeated_section_re = re.compile(
            rf"^(?:Section|Article|Chapter|Part)\s+{identifier}\b",
            re.IGNORECASE
        )
        any_subsection_re = re.compile(
            rf"^(?:(?:Section|Article|Chapter|Part)\s+)?{identifier}\.\d+(?:\.\d+)*\b",
            re.IGNORECASE
        )

        summaries = []
        seen = set()
        index = 0

        while index < len(lines):
            match = child_re.match(lines[index])
            if not match:
                index += 1
                continue

            title = (match.group(2) or "").strip(" :-–—")
            content_start = index + 1

            if not title and content_start < len(lines):
                candidate = lines[content_start]
                if (
                    candidate
                    and len(candidate) <= 100
                    and not any_subsection_re.match(candidate)
                    and not repeated_section_re.match(candidate)
                    and not candidate.endswith((".", "!", "?"))
                ):
                    title = candidate
                    content_start += 1

            if not title:
                index += 1
                continue

            content_lines = []
            cursor = content_start

            while cursor < len(lines):
                candidate = lines[cursor]
                if child_re.match(candidate) or repeated_section_re.match(candidate):
                    break
                if any_subsection_re.match(candidate):
                    break
                content_lines.append(candidate)
                cursor += 1

            raw_content = re.sub(
                r"\s+",
                " ",
                " ".join(content_lines)
            ).strip()

            colon_index = raw_content.find(":")
            if (
                0 < colon_index <= 220
                and not re.search(r"[.!?]", raw_content[:colon_index])
            ):
                key_text = raw_content[:colon_index].strip()
                key_text = re.sub(
                    r",?\s+for\s+example$",
                    "",
                    key_text,
                    flags=re.IGNORECASE
                ).strip()
                if key_text:
                    key_text += "."
            else:
                key_text = self._limit_grounded_explanation_text(
                    raw_content,
                    max_sentences=1,
                    max_chars=240
                )

            normalized = re.sub(r"\s+", " ", title).strip().casefold()

            if normalized not in seen:
                seen.add(normalized)
                summaries.append((
                    re.sub(r"\s+", " ", title).strip(),
                    key_text
                ))

            if len(summaries) >= max_topics:
                break

            index = max(cursor, index + 1)

        return summaries

    def _section_body_key_points_from_context(
        self,
        context: str,
        reference,
        max_points: int = 3
    ):

        """Extract a few informative bullet points from one exact Section.

        Some source PDFs repeat the parent ``Section N`` heading between
        child subsection titles, so a title-based parser can legitimately
        find ``N.1`` / ``N.2`` while their explanatory text appears in
        separate bullet blocks. This fallback stays inside the already
        accepted exact Section context and only returns source-derived text.
        """

        if (
            not context
            or reference is None
            or not reference.is_section_like
        ):
            return []

        lines = [
            line.strip()
            for line in context.splitlines()
            if line.strip()
            and not line.strip().startswith("=====")
        ]

        identifier = re.escape(reference.identifier)
        parent_re = re.compile(
            rf"^(?:Section\s+)?{identifier}\b",
            re.IGNORECASE
        )
        child_re = re.compile(
            rf"^{identifier}\.\d+(?:\.\d+)*(?:\s+.*)?$",
            re.IGNORECASE
        )

        points = []
        seen = set()
        index = 0

        while index < len(lines):
            line = lines[index]
            standalone_bullet = line in {"-", "•"}
            inline_bullet = bool(re.match(r"^[-•]\s+\S", line))

            if not (standalone_bullet or inline_bullet):
                index += 1
                continue

            if standalone_bullet:
                parts = []
                cursor = index + 1
            else:
                parts = [re.sub(r"^[-•]\s+", "", line).strip()]
                cursor = index + 1

            while cursor < len(lines):
                candidate = lines[cursor]
                if (
                    candidate in {"-", "•"}
                    or re.match(r"^[-•]\s+\S", candidate)
                    or parent_re.match(candidate)
                    or child_re.match(candidate)
                ):
                    break
                parts.append(candidate)
                cursor += 1

            point = re.sub(r"\s+", " ", " ".join(parts)).strip()
            point = self._limit_grounded_explanation_text(
                point,
                max_sentences=1,
                max_chars=320
            )

            normalized = self._normalize_identity_match_text(point)
            word_count = len(point.split())

            # Skip tiny labels, cross-reference-only lines, and obvious code
            # fragments. Long natural-language bullets that mention code
            # constructs remain eligible.
            looks_like_code = (
                word_count < 5
                or re.match(r"^(?:see also|references?|citations?)\b", point, re.IGNORECASE)
                or (
                    point.count(";") >= 2
                    and point.count("{") + point.count("}") >= 1
                )
            )

            if (
                point
                and not looks_like_code
                and normalized
                and normalized not in seen
            ):
                seen.add(normalized)
                points.append(point)

                if len(points) >= max_points:
                    break

            index = max(cursor, index + 1)

        return points

    def _structured_section_explanation_from_context(
        self,
        context: str,
        reference,
        current_answer: str = ""
    ):

        """Build a safe fallback when a model explains only a Section title."""

        clean_answer = re.sub(
            r"\s+",
            " ",
            (current_answer or "")
        ).strip()

        # Preserve a model answer that is already a substantive explanation.
        if (
            len(clean_answer) >= 140
            and (
                len(re.findall(r"[.!?](?:\s|$)", clean_answer)) >= 2
                or "\n- " in (current_answer or "")
            )
        ):
            return current_answer

        title, intro = self._section_heading_and_intro_from_context(
            context,
            reference
        )
        topics = self._section_subtopic_titles_from_context(
            context,
            reference
        )
        topic_summaries = self._section_subtopic_summaries_from_context(
            context,
            reference
        )
        body_key_points = self._section_body_key_points_from_context(
            context,
            reference
        )

        if not title and not intro and not topics and not body_key_points:
            return current_answer

        pieces = []
        display = f"Section {reference.identifier}"

        if title:
            pieces.append(f"{display}: {title.rstrip('.')}")
        elif clean_answer:
            pieces.append(clean_answer.rstrip("."))
        else:
            pieces.append(display)

        if intro:
            pieces.append(intro.rstrip())

        if topic_summaries:
            detail_items = []
            substantive_summary_count = 0

            for topic_title, key_text in topic_summaries:
                cleaned_key_text = (key_text or "").strip()
                if (
                    cleaned_key_text.endswith(".")
                    and not cleaned_key_text.endswith("...")
                ):
                    cleaned_key_text = cleaned_key_text[:-1]

                if cleaned_key_text:
                    substantive_summary_count += 1
                    detail_items.append(
                        f"- {topic_title}: {cleaned_key_text}"
                    )
                else:
                    detail_items.append(f"- {topic_title}")

            pieces.append(
                "Key points:\n"
                + "\n".join(detail_items)
            )

            # If child titles were found but their source layout separates
            # the actual explanatory bullets from those titles, add a few
            # exact-context details instead of returning a title-only
            # "explanation".
            if substantive_summary_count == 0 and body_key_points:
                pieces.append(
                    "Supported details:\n"
                    + "\n".join(
                        f"- {point}"
                        for point in body_key_points
                    )
                )
        elif topics:
            if len(topics) == 1:
                topic_text = topics[0]
            elif len(topics) == 2:
                topic_text = f"{topics[0]} and {topics[1]}"
            else:
                topic_text = ", ".join(topics[:-1]) + f", and {topics[-1]}"

            pieces.append(
                "The section covers topics including "
                + topic_text
                + "."
            )

            if body_key_points:
                pieces.append(
                    "Supported details:\n"
                    + "\n".join(
                        f"- {point}"
                        for point in body_key_points
                    )
                )
        elif body_key_points:
            pieces.append(
                "Supported details:\n"
                + "\n".join(
                    f"- {point}"
                    for point in body_key_points
                )
            )

        prose_pieces = []
        bullet_pieces = []

        for piece in pieces:
            if not piece:
                continue
            if "\n- " in piece:
                bullet_pieces.append(piece.strip())
            else:
                prose_pieces.append(piece.rstrip("."))

        answer = ". ".join(prose_pieces).strip()

        if answer and not answer.endswith("."):
            answer += "."

        if bullet_pieces:
            answer = (
                answer
                + "\n\n"
                + "\n\n".join(bullet_pieces)
            ).strip()

        return answer or current_answer

    def _compact_label_value_pairs_from_context(
        self,
        context: str,
        max_chars: int = 1600
    ):

        """Find compact label/value facts such as policy eligibility fields."""

        if not context or len(context) > max_chars:
            return []

        lines = [
            line.strip()
            for line in context.splitlines()
            if line.strip()
            and not line.strip().startswith("=====")
        ]

        def looks_like_label(value: str) -> bool:
            if not value or len(value) > 70:
                return False
            if value.endswith((".", "!", "?", ";")):
                return False
            if re.match(r"^(?:[-*•]|\d+[.)])\s+", value):
                return False
            words = value.split()
            return 1 <= len(words) <= 8

        pairs = []

        for index in range(len(lines) - 1):
            label = lines[index]
            value = lines[index + 1]

            if not looks_like_label(label):
                continue
            if looks_like_label(value):
                continue
            if len(value.split()) < 4:
                continue

            pairs.append((label, value))

        return pairs

    def _grounded_compact_labeled_overview(
        self,
        context: str,
        question: str,
    ):
        """Return all compact labeled facts for a broad topic overview.

        The method is domain-neutral and runs only when the user asks broadly
        about a topic/policy and the accepted context exposes a small structured
        set of explicit label/value facts.  It prevents a small model from
        silently dropping one nearby field (for example eligibility) while never
        inventing fields that are absent from the source.
        """

        if not context or not question:
            return ""

        clean = re.sub(r"\s+", " ", str(question or "").strip().casefold())
        broad = bool(re.search(
            r"^(?:tell\s+me\s+about|explain|describe|summari[sz]e|give\s+me\s+(?:an?\s+)?overview|"
            r"ano\s+(?:ba|ang)|paki[- ]?explain|ipaliwanag)\b",
            clean,
            re.IGNORECASE,
        )) or bool(re.search(r"\b(?:overview|summary|short\s+summary)\b", clean))
        if not broad:
            return ""

        # A question that explicitly requests one relation should stay on its
        # dedicated relation path rather than receiving the whole compact block.
        if re.match(r"^(?:who|sino|how\s+many|how\s+much|when|where|kailan|saan)\b", clean):
            return ""

        pairs = self._compact_label_value_pairs_from_context(context)
        if not (3 <= len(pairs) <= 8):
            return ""

        rendered = []
        seen = set()
        for label, value in pairs:
            label_clean = re.sub(r"\s+", " ", str(label or "")).strip()
            value_clean = re.sub(r"\s+", " ", str(value or "")).strip()
            key = label_clean.casefold()
            if not label_clean or not value_clean or key in seen:
                continue
            seen.add(key)
            rendered.append(f"- **{label_clean}:** {value_clean}")

        return "\n".join(rendered) if len(rendered) >= 3 else ""

    def _grounded_compact_compound_answer(
        self,
        context: str,
        question: str,
    ):

        """Return a deterministic answer for narrow compact multi-part facts.

        This recovery is intentionally conservative. It only operates on
        compact label/value sources (for example a short policy table/text
        block) and only when every requested facet can be matched to an
        explicit source value. No outside knowledge, inference, or LLM call is
        used. If any facet is ambiguous, the normal grounded generation path
        remains in control.
        """

        if not context or not question:
            return ""

        facets = self._compound_facets_for_answer_check(question)
        pairs = self._compact_label_value_pairs_from_context(context)

        if len(facets) < 2 or len(pairs) < 2:
            return ""

        normalized_pairs = []
        for label, value in pairs:
            label_norm = self._normalize_identity_match_text(label)
            value_norm = self._normalize_identity_match_text(value)
            normalized_pairs.append((label, value, label_norm, value_norm))

        generic_tokens = {
            "how", "much", "many", "what", "which", "who",
            "is", "are", "was", "were", "be", "been", "the",
            "a", "an", "and", "at", "of", "to", "for",
            "provided", "provide", "provides", "must", "can",
            "may", "request", "requests", "employee", "employees",
            "policy", "policies", "it", "this", "that", "listed",
            "above", "ang", "ano", "anong", "sino", "ilan",
        }

        def subject_tokens(facet_norm: str):
            return [
                token
                for token in facet_norm.split()
                if len(token) >= 3 and token not in generic_tokens
            ]

        def best_pair_for_subject(facet_norm: str, require_number=False):
            tokens = subject_tokens(facet_norm)
            best = None
            best_score = 0

            for item in normalized_pairs:
                label, value, label_norm, value_norm = item
                if require_number and not re.search(r"\b\d+(?:\.\d+)?\b", value):
                    continue

                score = 0
                for token in set(tokens):
                    if re.search(rf"\b{re.escape(token)}\b", label_norm):
                        score += 3
                    elif re.search(rf"\b{re.escape(token)}\b", value_norm):
                        score += 1

                if score > best_score:
                    best_score = score
                    best = item

            return best if best_score >= 2 else None

        answers = []

        for facet in facets:
            facet_norm = self._normalize_identity_match_text(facet)
            matched = None

            quantity_intent = bool(
                re.search(r"\bhow\s+(?:many|much)\b", facet_norm)
                or re.search(r"\bilan\b", facet_norm)
            )
            approver_intent = bool(
                re.search(
                    r"\b(?:approve|approves|approval|approver|authorize|authorizes)\b",
                    facet_norm,
                )
            )
            eligibility_intent = bool(
                re.search(r"\b(?:eligible|eligibility|qualified|qualification)\b", facet_norm)
            )
            entitlement_intent = bool(
                re.search(r"\b(?:entitled|entitlement)\b", facet_norm)
            )

            if quantity_intent:
                matched = best_pair_for_subject(facet_norm, require_number=True)

            elif approver_intent:
                for item in normalized_pairs:
                    _, value, label_norm, value_norm = item
                    if (
                        "approval" in label_norm
                        or "approver" in label_norm
                        or re.search(r"\b(?:approve|approval|approver)\b", value_norm)
                    ):
                        matched = item
                        break

            elif eligibility_intent:
                for item in normalized_pairs:
                    _, value, label_norm, value_norm = item
                    if (
                        "eligibility" in label_norm
                        or re.search(r"\b(?:eligible|qualified|regular employee|regular employees)\b", value_norm)
                    ):
                        matched = item
                        break

            elif entitlement_intent:
                matched = best_pair_for_subject(facet_norm, require_number=False)

            if matched is None:
                return ""

            value = re.sub(r"\s+", " ", matched[1]).strip()
            if not value:
                return ""
            if not value.endswith((".", "!", "?")):
                value += "."
            answers.append(value)

        # Every requested facet must have its own explicit source-backed item.
        if len(answers) != len(facets):
            return ""

        deduped = []
        seen = set()
        for answer in answers:
            key = self._normalize_identity_match_text(answer)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(answer)

        if len(deduped) != len(facets):
            return ""

        return "\n".join(f"- {answer}" for answer in deduped)


    def _compact_pair_is_covered(
        self,
        answer: str,
        label: str,
        value: str
    ):

        normalized_answer = re.sub(
            r"[^a-z0-9\s]",
            " ",
            (answer or "").lower()
        )
        normalized_answer = re.sub(r"\s+", " ", normalized_answer).strip()

        label_tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", label.lower())
            if token not in {"the", "a", "an", "of", "and", "or", "leave", "policy"}
            and len(token) >= 3
        ]

        if any(
            re.search(rf"\b{re.escape(token)}\b", normalized_answer)
            for token in label_tokens
        ):
            return True

        value_stopwords = {
            "the", "a", "an", "of", "and", "or", "to", "is",
            "are", "all", "this", "that", "employee", "employees",
            "leave", "policy", "listed", "above", "provides", "provided"
        }
        value_tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", value.lower())
            if token not in value_stopwords
            and len(token) >= 4
        ]

        matches = sum(
            1
            for token in set(value_tokens)
            if re.search(rf"\b{re.escape(token)}\b", normalized_answer)
        )

        return matches >= min(2, len(set(value_tokens))) if value_tokens else False

    def _apply_grounded_explanation_coverage(
        self,
        context: str,
        answer_focus: str,
        current_answer: str
    ):

        """Add only omitted explicit facts from compact structured sources."""

        if not answer_focus.startswith("GROUNDED EXPLANATION:"):
            return current_answer

        pairs = self._compact_label_value_pairs_from_context(context)

        if len(pairs) < 3:
            return current_answer

        missing = [
            (label, value)
            for label, value in pairs
            if not self._compact_pair_is_covered(
                current_answer,
                label,
                value
            )
        ]

        if not missing:
            return current_answer

        additions = [
            f"{label}: {value}"
            for label, value in missing[:3]
        ]

        base = (current_answer or "").strip()

        if base and not base.endswith((".", "!", "?")):
            base += "."

        return " ".join(
            part
            for part in [base] + additions
            if part
        ).strip()

    def _structured_section_detail_from_context(
        self,
        context: str,
        reference,
        question: str = "",
        resolved_question: str = "",
    ):

        """Extract the closest supported passage for a named Section subtopic.

        This stays entirely inside the exact structured Section block. It is
        intentionally lexical and conservative: if the user's meaningful
        subtopic words cannot be found locally, generation keeps the existing
        fallback path instead of inventing a Section detail.
        """

        if not context or reference is None or not reference.is_section_like:
            return ""

        raw_query = re.sub(
            r"\b(?:section|article|chapter|part)\s+" + re.escape(reference.identifier) + r"\b",
            " ",
            str(resolved_question or question or ""),
            flags=re.IGNORECASE,
        )
        normalized_query = self._normalize_identity_match_text(raw_query)
        stop = {
            "what", "does", "say", "says", "about", "regarding", "concerning",
            "the", "a", "an", "of", "to", "for", "in", "on", "with", "and",
            "ano", "ang", "sinasabi", "tungkol", "sa", "ng", "mga", "ayon",
            "give", "state", "explain", "describe", "detail", "details",
        }
        tokens = [
            token for token in normalized_query.split()
            if len(token) >= 3 and token not in stop and not token.isdigit()
        ]
        # Preserve order while deduplicating.
        tokens = list(dict.fromkeys(tokens))[:10]
        if not tokens:
            return ""

        variant_map = {
            "decidability": {"decidability", "decidable", "undecidable"},
            "decidable": {"decidability", "decidable", "undecidable"},
            "undecidable": {"decidability", "decidable", "undecidable"},
            "category": {"category", "categories"},
            "categories": {"category", "categories"},
            "applicability": {"applicability", "applicable", "applies"},
            "scope": {"scope", "scopes"},
        }
        token_variants = {
            token: variant_map.get(token, {token})
            for token in tokens
        }

        lines = [
            line.strip()
            for line in str(context).splitlines()
            if line.strip() and not line.strip().startswith("=====")
        ]
        if not lines:
            return ""

        # Prefer a child subsection heading whose title directly matches the
        # requested subtopic. This avoids selecting schema/presentation lines
        # such as ``Decidability, Scope`` when the same exact Section also
        # contains a real child heading like ``6.5 Decidability of rules``.
        child_number_re = re.compile(
            rf"^(?:(?:Section|Article|Chapter|Part)\s+)?{re.escape(reference.identifier)}(?:\.\d+)+(?:\s+.*)?$",
            re.IGNORECASE,
        )
        query_phrase = " ".join(tokens)
        heading_candidates = []
        for index, line in enumerate(lines):
            if not child_number_re.match(line):
                continue
            heading_text = line
            if re.match(
                rf"^(?:(?:Section|Article|Chapter|Part)\s+)?{re.escape(reference.identifier)}(?:\.\d+)+$",
                line,
                flags=re.IGNORECASE,
            ) and index + 1 < len(lines):
                heading_text = f"{line} {lines[index + 1]}"
            heading_norm = self._normalize_identity_match_text(heading_text)
            hit_count = sum(
                any(variant in heading_norm for variant in token_variants[token])
                for token in tokens
            )
            if hit_count:
                phrase_bonus = 2.0 if query_phrase and query_phrase in heading_norm else 0.0
                heading_candidates.append((float(hit_count) + phrase_bonus, index))

        if heading_candidates:
            heading_candidates.sort(key=lambda item: item[0], reverse=True)
            _, start = heading_candidates[0]
            collected = []
            for index in range(start, len(lines)):
                line = lines[index]
                if index > start and child_number_re.match(line):
                    break
                collected.append(line)
                if len(" ".join(collected)) >= 1500:
                    break
            detail = re.sub(r"\s+", " ", " ".join(collected)).strip()
            if detail:
                return self._limit_grounded_explanation_text(
                    detail,
                    max_sentences=6,
                    max_chars=1600,
                )

        best_index = -1
        best_score = 0.0

        for index, line in enumerate(lines):
            norm = self._normalize_identity_match_text(line)
            if not norm:
                continue
            hit_count = sum(
                any(variant in norm for variant in token_variants[token])
                for token in tokens
            )
            if not hit_count:
                continue
            score = float(hit_count)
            # Prefer explanatory prose over compact schema labels when both
            # contain the same requested concept.
            if len(line.split()) >= 8:
                score += 0.60
            if re.search(
                r"\b(?:classified\s+as|means|refers\s+to|describes|is\s+.+?\s+if|are\s+.+?\s+if)\b",
                norm,
                flags=re.IGNORECASE,
            ):
                score += 0.75
            if "," in line and len(line.split()) <= 5 and not re.search(
                r"\b(?:is|are|was|were|means|describes|defines)\b",
                norm,
            ):
                score -= 0.50
            # Exact-phrase bonus is useful for multi-word subtopics, but a
            # single generic field word (for example ``decidability``) can
            # otherwise over-promote compact schema/presentation rows over
            # the nearby explanatory prose that actually answers the user.
            if len(tokens) >= 2 and query_phrase and query_phrase in norm:
                score += 3.0
            # Prefer subsection-like headings and compact lines because they
            # usually bind the following paragraph to the requested subtopic.
            if re.match(
                rf"^(?:(?:section|article|chapter|part)\s+)?{re.escape(reference.identifier)}(?:\.\d+)+\b",
                line,
                flags=re.IGNORECASE,
            ):
                score += 1.5
            if len(line) <= 140:
                score += 0.25
            if score > best_score:
                best_score = score
                best_index = index

        required_hits = 1 if len(tokens) <= 2 else 2
        if best_index < 0 or best_score < required_hits:
            return ""

        child_heading = re.compile(
            rf"^(?:(?:Section|Article|Chapter|Part)\s+)?{re.escape(reference.identifier)}(?:\.\d+)+(?:\s+.*)?$",
            re.IGNORECASE,
        )

        start = best_index
        # If the hit is body text, include the closest preceding child heading
        # when it is nearby so the returned passage remains self-describing.
        # If the best hit is already a child heading, never walk backward into
        # the previous subsection.
        if not child_heading.match(lines[best_index]):
            for candidate in range(best_index - 1, max(-1, best_index - 4), -1):
                if child_heading.match(lines[candidate]):
                    start = candidate
                    break

        collected = []
        for index in range(start, len(lines)):
            line = lines[index]
            if index > start and child_heading.match(line):
                break
            collected.append(line)
            if len(" ".join(collected)) >= 1100:
                break

        detail = re.sub(r"\s+", " ", " ".join(collected)).strip()
        if not detail:
            return ""

        distinction_query = bool(
            re.search(r"\brules?\b", normalized_query)
            and re.search(r"\bdirectives?\b", normalized_query)
        )
        return self._limit_grounded_explanation_text(
            detail,
            max_sentences=8 if distinction_query else 4,
            max_chars=1800 if distinction_query else 1200,
        )


    def _structured_section_topics_from_context(self, context: str, reference):
        """Return concise direct child headings from one exact structured section."""
        if not context or reference is None or not reference.is_section_like:
            return ""

        topics = self._section_subtopic_titles_from_context(
            context=context,
            reference=reference,
            max_topics=12,
        )
        if not topics:
            return ""
        return "\n".join(f"- {topic}" for topic in topics)

    def _grounded_compact_comparison_answer(self, context: str, question: str):
        """Return compact numeric comparisons when both labeled values are explicit."""
        if not context or not question:
            return ""

        # Date-order questions require event/date binding, not generic
        # adjacent label/value extraction. Treat them separately so years in
        # prose cannot masquerade as numeric comparison values.
        if re.search(
            r"\b(?:before|after|later|earlier|nauna|sumunod|kaysa)\b",
            str(question),
            flags=re.IGNORECASE,
        ):
            return ""

        lines = [
            line.strip()
            for line in context.splitlines()
            if line.strip() and not line.strip().startswith("=====")
        ]
        pairs = []
        for index in range(len(lines) - 1):
            label = lines[index]
            value = lines[index + 1]
            if len(label) > 80 or len(label.split()) > 8:
                continue
            if not re.search(r"\b\d+(?:\.\d+)?\b", value):
                continue
            pairs.append((label, value))

        if len(pairs) < 2:
            return ""

        qnorm = self._normalize_identity_match_text(question)
        generic = {
            "compare", "comparison", "contrast", "difference", "different",
            "annual", "yearly", "amount", "amounts", "credits", "credit",
            "allowance", "allowances", "ihambing", "taunang", "ang", "at",
        }
        matches = []
        for label, value in pairs:
            lnorm = self._normalize_identity_match_text(label)
            tokens = [t for t in lnorm.split() if len(t) >= 3 and t not in generic]
            if not tokens:
                continue
            score = sum(
                1 for token in set(tokens)
                if re.search(rf"\b{re.escape(token)}\b", qnorm)
            )
            if score > 0:
                matches.append((score, label, re.sub(r"\s+", " ", value).strip()))

        matches.sort(key=lambda item: item[0], reverse=True)
        selected = []
        seen_labels = set()
        for _, label, value in matches:
            key = self._normalize_identity_match_text(label)
            if key in seen_labels:
                continue
            seen_labels.add(key)
            selected.append((label, value))
            if len(selected) == 2:
                break

        if len(selected) != 2:
            return ""
        return "\n".join(f"- {label}: {value}" for label, value in selected)

    def _grounded_temporal_bm25_retry(
        self,
        context: str,
        results,
        question: str,
        semantic_target_question: str = "",
    ):
        """Supplement temporal grounding with BM25-only facet evidence.

        This retry is intentionally *not* another generation/retrieval path.
        It runs only after accepted context could not deterministically bind
        both requested event dates.  Candidate chunks come from the already
        indexed BM25 corpus and are used only to attempt the strict temporal
        finalizer.  They are never exposed to the LLM unless the deterministic
        two-date comparison succeeds.
        """
        if not question or not self._is_explicit_temporal_comparison_question(question):
            return "", context, list(results or [])

        try:
            retriever = self.query_service._get_retriever()
        except Exception:
            return "", context, list(results or [])

        extractor = getattr(retriever, "_extract_compound_facets", None)
        bm25 = getattr(retriever, "bm25", None)
        search = getattr(bm25, "search", None)
        if not callable(extractor) or not callable(search):
            return "", context, list(results or [])

        facet_question = semantic_target_question or question
        facets = extractor(facet_question)
        if len(facets) != 2 and facet_question != question:
            facets = extractor(question)
        if len(facets) != 2:
            return "", context, list(results or [])

        existing = list(results or [])
        supplements = []
        seen = set()

        def identity(item):
            metadata = item.get("metadata", {}) or {}
            return (
                metadata.get("file_path") or metadata.get("file_name") or "Unknown",
                str(metadata.get("chunk_id", "")),
                str(item.get("text", ""))[:120],
            )

        for item in existing:
            seen.add(identity(item))

        coverage_search = getattr(bm25, "coverage_search", None)

        for facet in facets:
            facet_candidates = list(search(facet, top_k=8) or [])
            if callable(coverage_search):
                try:
                    facet_candidates.extend(
                        coverage_search(facet, top_k=8, minimum_score=0.26) or []
                    )
                except Exception:
                    pass

            for item in facet_candidates:
                # Standard BM25 uses positive raw scores; coverage_search uses
                # a 0..1 lexical coverage score. Both are discovery signals
                # only—the temporal answer still needs strict date/event
                # binding below.
                if float(item.get("score", 0.0) or 0.0) <= 0.0:
                    continue
                key = identity(item)
                if key in seen:
                    continue
                seen.add(key)
                supplements.append(item)
                if len(supplements) >= 10:
                    break

        # Do not stop when top-K discovery yields no *new* supplement. The
        # record-scan fallback below is specifically designed for that case.
        pieces = [str(context or "").strip()] if str(context or "").strip() else []
        start_index = len(existing) + 1
        for offset, item in enumerate(supplements, start=start_index):
            text = str(item.get("text", "") or "").strip()
            if text:
                pieces.append(f"===== DOCUMENT {offset} =====\n{text}")
        retry_context = "\n\n".join(pieces).strip()

        answer = self._grounded_temporal_comparison_answer(retry_context, question)

        # Last deterministic recovery: top-K BM25 can still miss an exact-date
        # chunk in a large heterogeneous corpus. Scan the *already loaded* BM25
        # records for full-date rows whose local text has strong lexical overlap
        # with each requested event facet. This uses no embedding/reranker/LLM
        # and creates no new index. It only supplies candidate source text to
        # the same strict temporal finalizer above.
        if not answer:
            records = list(getattr(bm25, "records", None) or [])
            if records:
                month_aliases = {
                    "enero": "january", "pebrero": "february", "marso": "march",
                    "abril": "april", "mayo": "may", "hunyo": "june",
                    "hulyo": "july", "agosto": "august", "setyembre": "september",
                    "oktubre": "october", "nobyembre": "november", "disyembre": "december",
                }
                relation_aliases = {
                    "deklarasyon": "declaration", "idineklara": "declaration",
                    "ipinahayag": "declaration", "proclamation": "declaration",
                    "proclaimed": "declaration", "declared": "declaration",
                    "pagpirma": "sign", "pinirmahan": "sign", "nilagdaan": "sign",
                    "lumagda": "sign", "signed": "sign", "signing": "sign",
                    "kalayaan": "independence", "kasarinlan": "independence",
                    "kasunduan": "treaty",
                }
                stop = {
                    "ang", "ng", "mga", "sa", "ay", "at", "na", "nang", "para",
                    "noong", "ba", "kaysa", "nauna", "sumunod", "the", "a", "an",
                    "of", "to", "for", "in", "on", "at", "and", "or", "was", "is",
                    "did", "before", "after", "earlier", "later",
                }
                full_date_re = re.compile(
                    r"\b(?:January|February|March|April|May|June|July|August|"
                    r"September|October|November|December)\s+\d{1,2},\s*(?:18|19|20)\d{2}\b",
                    flags=re.IGNORECASE,
                )

                def canonical_tokens(value):
                    norm = self._normalize_identity_match_text(value)
                    output = []
                    for token in norm.split():
                        token = month_aliases.get(token, token)
                        token = relation_aliases.get(token, token)
                        if len(token) < 3 or token in stop:
                            continue
                        output.append(token)
                    return list(dict.fromkeys(output))

                record_candidates = []
                for facet_index, facet in enumerate(facets):
                    facet_tokens = canonical_tokens(facet)
                    if not facet_tokens:
                        continue
                    facet_norm = " ".join(facet_tokens)
                    explicit_years = set(re.findall(r"\b(?:18|19|20)\d{2}\b", facet_norm))
                    explicit_months = {
                        month for month in (
                            "january", "february", "march", "april", "may", "june",
                            "july", "august", "september", "october", "november", "december",
                        ) if month in facet_tokens
                    }
                    scored = []
                    for record in records:
                        text = str(record.get("text", "") or "")
                        if not text or not full_date_re.search(text):
                            continue
                        text_norm = " ".join(canonical_tokens(text))
                        text_words = set(text_norm.split())
                        if explicit_years and not explicit_years.intersection(text_words):
                            continue
                        if explicit_months and not explicit_months.intersection(text_words):
                            continue
                        overlap = set(facet_tokens).intersection(text_words)
                        # Require multiple independent anchors so generic dates
                        # from unrelated timeline material cannot qualify.
                        required = 2 if len(facet_tokens) <= 4 else 3
                        if len(overlap) < min(required, len(set(facet_tokens))):
                            continue
                        score = len(overlap) / max(1, len(set(facet_tokens)))
                        if explicit_years:
                            score += 0.35
                        if explicit_months:
                            score += 0.25
                        metadata = record.get("metadata", {}) or {}
                        title_norm = self._normalize_identity_match_text(
                            f"{metadata.get('file_name', '')} {metadata.get('section_title', '')}"
                        )
                        if any(token in title_norm for token in overlap):
                            score += 0.10
                        scored.append((score, record))

                    scored.sort(key=lambda row: row[0], reverse=True)
                    for score, record in scored[:4]:
                        item = {
                            "text": record.get("text", ""),
                            "metadata": record.get("metadata", {}) or {},
                            "score": float(score),
                            "_temporal_record_scan": True,
                            "_temporal_facet_index": facet_index,
                        }
                        key = identity(item)
                        if key in seen:
                            continue
                        seen.add(key)
                        record_candidates.append(item)

                if record_candidates:
                    supplements.extend(record_candidates)
                    pieces = [str(context or "").strip()] if str(context or "").strip() else []
                    start_index = len(existing) + 1
                    for offset, item in enumerate(supplements, start=start_index):
                        text = str(item.get("text", "") or "").strip()
                        if text:
                            pieces.append(f"===== DOCUMENT {offset} =====\n{text}")
                    retry_context = "\n\n".join(pieces).strip()
                    answer = self._grounded_temporal_comparison_answer(retry_context, question)
                    if answer:
                        evidence_logger.record_event(
                            event_name="TEMPORAL GROUNDING RETRY",
                            status="BM25 RECORD SCAN ACCEPTED",
                            details={
                                "facets": facets,
                                "record_scan_candidates": len(record_candidates),
                                "policy": (
                                    "Already-loaded BM25 records were scanned for full-date + "
                                    "event-anchor overlap, then passed through strict deterministic "
                                    "two-date binding only."
                                ),
                            },
                        )

        if not answer:
            return "", context, existing

        # Retain only supplements that contain a date actually used by the
        # deterministic answer, keeping source chips concise and traceable.
        used_dates = set(re.findall(
            r"\b(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{1,2},\s*(?:18|19|20)\d{2}\b",
            answer,
            flags=re.IGNORECASE,
        ))
        selected_supplements = []
        for item in supplements:
            text = str(item.get("text", "") or "")
            if not used_dates or any(date.lower() in text.lower() for date in used_dates):
                selected_supplements.append(item)

        merged_results = existing + selected_supplements
        evidence_logger.record_event(
            event_name="TEMPORAL GROUNDING RETRY",
            status="BM25 FACET EVIDENCE ACCEPTED",
            details={
                "facets": facets,
                "supplemental_chunks": len(selected_supplements),
                "policy": (
                    "BM25-only indexed evidence was used solely for strict "
                    "deterministic two-date binding; it was not sent to the LLM."
                ),
            },
        )
        return answer, retry_context, merged_results

    def _grounded_temporal_comparison_answer(self, context: str, question: str):
        """Use the locked Cycle-1 temporal finalizer first, then generic extensions.

        The certified v6.4.14.2.4 logic remains authoritative for the wording
        it already understands.  Broader compare/chronological/date-layout
        support is strictly fallback-only so new generality cannot change a
        previously certified temporal answer.
        """
        certified = self._grounded_certified_temporal_comparison_answer(
            context, question
        )
        if certified:
            return certified
        return self._grounded_generalized_temporal_comparison_answer(
            context, question
        )

    def _grounded_certified_temporal_comparison_answer(self, context: str, question: str):
        """Answer explicit temporal comparisons only when both events bind to grounded dates.

        The binding is intentionally stricter than generic proximity matching.  A date
        must satisfy any explicit month/year cue in its own facet and its local source
        window must match the facet's event anchors.  This prevents unrelated timeline
        dates from being compared merely because they occur in the same retrieved
        document.
        """

        if not context or not question:
            return ""

        clean_question = re.sub(r"\s+", " ", str(question)).strip(" .?!")
        if not re.search(
            r"\b(?:before|after|later|earlier|nauna|sumunod|kaysa)\b",
            clean_question,
            flags=re.IGNORECASE,
        ):
            return ""

        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }
        month_aliases = {
            "enero": "january", "pebrero": "february", "marso": "march",
            "abril": "april", "mayo": "may", "hunyo": "june",
            "hulyo": "july", "agosto": "august", "setyembre": "september",
            "oktubre": "october", "nobyembre": "november", "disyembre": "december",
        }

        def canonical_event_text(value: str) -> str:
            normalized = self._normalize_identity_match_text(value)
            token_map = {
                "deklarasyon": "declare", "declaration": "declare",
                "declared": "declare", "declares": "declare",
                "proclamation": "declare", "proclaimed": "declare",
                "idineklara": "declare", "ipinahayag": "declare",
                "pagpirma": "sign", "pinirmahan": "sign",
                "nilagdaan": "sign", "lumagda": "sign",
                "signed": "sign", "signing": "sign", "signature": "sign",
                "signatures": "sign", "kalayaan": "independence",
                "kasarinlan": "independence", "kasarinlán": "independence",
                "kasunduan": "treaty",
            }
            tokens = []
            for token in normalized.split():
                mapped = month_aliases.get(token, token)
                mapped = token_map.get(mapped, mapped)
                tokens.append(mapped)
            return " ".join(tokens)

        def split_facets(value: str):
            # Tagalog yes/no ordering: "Nauna ba X kaysa Y?" / "Mas nauna ba..."
            match = re.match(
                r"^\s*(?:mas\s+)?(?:nauna|sumunod)\s+(?:ba\s+)?(.+?)\s+kaysa\s+(.+?)\s*$",
                value,
                flags=re.IGNORECASE,
            )
            if match:
                return [match.group(1).strip(), match.group(2).strip()]

            # Tagalog comparative choice: "Alin ang nauna: X o Y?"
            match = re.match(
                r"^\s*alin\s+ang\s+(?:nauna|sumunod)\s*:\s*(.+?)\s+(?:o|or)\s+(.+?)\s*$",
                value,
                flags=re.IGNORECASE,
            )
            if match:
                return [match.group(1).strip(), match.group(2).strip()]

            # English yes/no ordering, including "before or after".
            match = re.match(
                r"^\s*(?:was|is|did)\s+(.+?)\s+"
                r"(?:before\s+or\s+after|before|after|earlier\s+than|later\s+than)\s+"
                r"(.+?)\s*$",
                value,
                flags=re.IGNORECASE,
            )
            if match:
                return [match.group(1).strip(), match.group(2).strip()]

            # Comparative choice: "Which came later: A or B?"
            match = re.match(
                r"^\s*which\s+came\s+(?:later|first|earlier)\s*:\s*(.+?)\s+or\s+(.+?)\s*$",
                value,
                flags=re.IGNORECASE,
            )
            if match:
                return [match.group(1).strip(), match.group(2).strip()]

            parts = re.split(
                r"\s+\bkaysa\b\s+|\s+\b(?:before\s+or\s+after|before|after|earlier\s+than|later\s+than)\b\s+",
                value,
                maxsplit=1,
                flags=re.IGNORECASE,
            )
            return [part.strip() for part in parts] if len(parts) == 2 else []

        facets = split_facets(clean_question)
        if len(facets) != 2:
            return ""

        text = str(context)
        full_date_pattern = re.compile(
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
            r"(\d{1,2}),\s*((?:18|19|20)\d{2})\b",
            flags=re.IGNORECASE,
        )
        date_matches = list(full_date_pattern.finditer(text))
        dates = []
        for match_index, match in enumerate(date_matches):
            month_name = match.group(1).lower()
            key = (int(match.group(3)), month_map[month_name], int(match.group(2)))

            # Never let event words leak across one retrieved document into a
            # date that belongs to another document. Within a document, bind the
            # date to the smallest useful local unit: usually its sentence, or
            # nearby wrapped lines for PDF/table layouts such as:
            #   Treaty of Paris / Signed / December 10, 1898.
            previous_document = text.rfind("===== DOCUMENT", 0, match.start())
            next_document = text.find("===== DOCUMENT", match.end())
            document_start = 0 if previous_document < 0 else previous_document
            document_end = len(text) if next_document < 0 else next_document

            # The accepted Windows evidence exposed a subtle table/prose
            # boundary case: an "Effective <date>" field immediately before
            # a sentence about a different signing date could inherit the next
            # event's words. Build a tight date-local anchor window and stop it
            # at neighboring full dates. This keeps event/date binding local
            # without changing retrieval, thresholds, or reranking.
            local_start = document_start
            for previous_match in reversed(date_matches[:match_index]):
                if previous_match.start() < document_start:
                    break
                local_start = max(local_start, previous_match.end())
                break

            local_end = document_end
            for next_match in date_matches[match_index + 1:]:
                if next_match.start() >= document_end:
                    break
                local_end = min(local_end, next_match.start())
                break

            previous_sentence = max(
                text.rfind(".", document_start, match.start()),
                text.rfind("?", document_start, match.start()),
                text.rfind("!", document_start, match.start()),
            )
            sentence_start = document_start if previous_sentence < 0 else previous_sentence + 1
            next_sentence_candidates = [
                index for index in (
                    text.find(".", match.end(), document_end),
                    text.find("?", match.end(), document_end),
                    text.find("!", match.end(), document_end),
                )
                if index >= 0
            ]
            sentence_end = (
                min(next_sentence_candidates) + 1
                if next_sentence_candidates
                else document_end
            )
            sentence_window = text[sentence_start:sentence_end]

            # For wrapped PDF/table text, event labels generally precede the
            # value (for example ``Signed`` -> ``December 10, 1898``).  Walk
            # backward only, and never cross a neighboring full date.  This
            # is the critical guard that prevents ``Effective April 11, 1899``
            # from borrowing ``Treaty ... signed`` words that introduce the
            # *next* December 10 date.
            date_line_start = text.rfind("\n", document_start, match.start()) + 1
            date_line_end = text.find("\n", match.end(), document_end)
            if date_line_end < 0:
                date_line_end = document_end

            line_start = max(local_start, date_line_start)
            for _ in range(6):
                prior_break = text.rfind(
                    "\n",
                    local_start,
                    max(local_start, line_start - 1),
                )
                if prior_break < local_start:
                    line_start = local_start
                    break
                line_start = prior_break + 1
            line_window = text[line_start:min(date_line_end, local_end)]

            # Normal prose should bind within its own sentence.  Standalone
            # date/value rows and long PDF-wrapped pseudo-sentences instead
            # use the backward structural window above.  Choosing one local
            # unit (rather than a symmetric character radius) avoids letting
            # the words that introduce the next event contaminate this date.
            date_line = text[date_line_start:date_line_end]
            date_line_remainder = full_date_pattern.sub("", date_line, count=1)
            date_is_standalone = not date_line_remainder.strip(" \t\r,;:-()[]")
            sentence_is_compact = len(sentence_window.strip()) <= 420
            anchor_window = (
                sentence_window
                if sentence_is_compact and not date_is_standalone
                else line_window
            )

            useful_windows = [
                value for value in (sentence_window, line_window)
                if value and len(value.strip()) >= len(match.group(0)) + 8
            ]
            window = min(useful_windows, key=len) if useful_windows else anchor_window

            dates.append({
                "key": key,
                "label": re.sub(r"\s+", " ", match.group(0)).strip(),
                "month": month_map[month_name],
                "year": int(match.group(3)),
                "window": canonical_event_text(window),
                "anchor_window": canonical_event_text(anchor_window),
            })

        if len({item["key"] for item in dates}) < 2:
            return ""

        stop = {
            "what", "when", "was", "were", "did", "is", "are", "the", "a", "an",
            "of", "to", "in", "on", "at", "noong", "ang", "ba", "pag", "sa", "ng",
            "mga", "than", "before", "after", "later", "earlier", "nauna", "sumunod",
            "kaysa", "date", "petsa", "which", "came", "or", "philippine", "philippines",
        }

        def facet_metadata(value: str):
            normalized = self._normalize_identity_match_text(value)
            raw_tokens = normalized.split()

            explicit_months = set()
            for token in raw_tokens:
                english_month = month_aliases.get(token, token)
                if english_month in month_map:
                    explicit_months.add(month_map[english_month])

            explicit_years = {
                int(year)
                for year in re.findall(r"\b(?:18|19|20)\d{2}\b", normalized)
            }

            canonical = canonical_event_text(value)
            tokens = []
            for token in canonical.split():
                if token in month_map or token in month_aliases:
                    continue
                if token in stop or token.isdigit() or len(token) < 3:
                    continue
                tokens.append(token)

            # Preserve order while deduplicating.  At least two anchors are
            # preferred when the facet provides them (e.g. sign+treaty,
            # declare+independence).
            unique_tokens = list(dict.fromkeys(tokens))
            return unique_tokens, explicit_months, explicit_years

        selected = []
        used_keys = set()
        for facet in facets:
            tokens, explicit_months, explicit_years = facet_metadata(facet)
            if not tokens:
                return ""

            ranked = []
            for item in dates:
                if item["key"] in used_keys:
                    continue
                if explicit_months and item["month"] not in explicit_months:
                    continue
                if explicit_years and item["year"] not in explicit_years:
                    continue

                # Score only anchors close to this exact date. The broader
                # sentence/line window remains available for diagnostics, but
                # it must not let a neighboring event borrow this date.
                anchor_tokens = set(item["anchor_window"].split())
                matched = [token for token in tokens if token in anchor_tokens]
                required_matches = 2 if len(tokens) >= 2 else 1
                if len(matched) < required_matches:
                    continue

                # Reward exact event-anchor coverage. Month/year constraints
                # are already hard filters above, not soft score bonuses. A
                # three-anchor event (for example action + object + qualifier)
                # naturally outranks a neighboring date that only shares two.
                score = (len(matched) * 10.0) + (len(matched) / max(1, len(tokens)))
                ranked.append((score, item, tuple(matched)))

            if not ranked:
                return ""

            ranked.sort(key=lambda row: row[0], reverse=True)
            best_score, best, _ = ranked[0]
            # If two different dates are equally plausible for the same event,
            # do not guess deterministically; let the normal grounded path deal
            # with it instead.
            if len(ranked) > 1 and ranked[1][0] == best_score and ranked[1][1]["key"] != best["key"]:
                return ""

            selected.append(best)
            used_keys.add(best["key"])

        first, second = selected
        if first["key"] == second["key"]:
            return ""

        earlier = first if first["key"] < second["key"] else second
        later = second if first["key"] < second["key"] else first
        first_is_earlier = first is earlier

        if re.match(
            r"^\s*(?:which\s+came\s+(?:later|first|earlier)|alin\s+ang\s+(?:nauna|sumunod))\b",
            clean_question,
            re.IGNORECASE,
        ):
            return f"{earlier['label']} came first; {later['label']} came later."

        if re.search(r"\b(?:nauna|before|earlier)\b", clean_question, re.IGNORECASE):
            if first_is_earlier:
                return f"Yes. {first['label']} came first; {second['label']} came later."
            return f"No. {second['label']} came first; {first['label']} came later."

        if re.search(r"\b(?:sumunod|after|later)\b", clean_question, re.IGNORECASE):
            if not first_is_earlier:
                return f"Yes. {first['label']} came later; {second['label']} came first."
            return f"No. {first['label']} came first; {second['label']} came later."

        return ""


    def _grounded_generalized_temporal_comparison_answer(self, context: str, question: str):
        """Answer explicit temporal comparisons only when both events bind to grounded dates.

        The binding is intentionally stricter than generic proximity matching.  A date
        must satisfy any explicit month/year cue in its own facet and its local source
        window must match the facet's event anchors.  This prevents unrelated timeline
        dates from being compared merely because they occur in the same retrieved
        document.
        """

        if not context or not question:
            return ""

        clean_question = re.sub(r"\s+", " ", str(question)).strip(" .?!")
        if not re.search(
            r"\b(?:before|after|later|earlier|nauna|sumunod|kaysa|"
            r"chronological|compare|first)\b",
            clean_question,
            flags=re.IGNORECASE,
        ):
            return ""

        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }
        month_aliases = {
            "enero": "january", "pebrero": "february", "marso": "march",
            "abril": "april", "mayo": "may", "hunyo": "june",
            "hulyo": "july", "agosto": "august", "setyembre": "september",
            "oktubre": "october", "nobyembre": "november", "disyembre": "december",
        }

        def canonical_event_text(value: str) -> str:
            normalized = self._normalize_identity_match_text(value)
            token_map = {
                "deklarasyon": "declare", "declaration": "declare",
                "declared": "declare", "declares": "declare",
                "proclamation": "declare", "proclaimed": "declare",
                "idineklara": "declare", "ipinahayag": "declare",
                "pagpirma": "sign", "pinirmahan": "sign",
                "nilagdaan": "sign", "lumagda": "sign",
                "signed": "sign", "signing": "sign", "signature": "sign",
                "signatures": "sign", "kalayaan": "independence",
                "kasarinlan": "independence", "kasarinlán": "independence",
                "kasunduan": "treaty",
                "execution": "execute", "executed": "execute",
                "execute": "execute", "establishment": "establish",
                "established": "establish", "establish": "establish",
                "founded": "establish", "founding": "establish",
                "capture": "capture", "captured": "capture",
                "surrender": "surrender", "surrendered": "surrender",
                "return": "return", "returned": "return",
                "proclaim": "declare", "proclaimed": "declare",
                "ended": "end",
                "ending": "end", "concluded": "end", "over": "end",
                "started": "start", "starting": "start", "began": "start",
                "begun": "start", "opening": "start",
            }
            tokens = []
            for token in normalized.split():
                mapped = month_aliases.get(token, token)
                mapped = token_map.get(mapped, mapped)
                tokens.append(mapped)
            return " ".join(tokens)

        def split_facets(value: str):
            # Tagalog yes/no ordering: "Nauna ba X kaysa Y?" / "Mas nauna ba..."
            match = re.match(
                r"^\s*(?:mas\s+)?(?:nauna|sumunod)\s+(?:ba\s+)?(.+?)\s+kaysa\s+(.+?)\s*$",
                value,
                flags=re.IGNORECASE,
            )
            if match:
                return [match.group(1).strip(), match.group(2).strip()]

            # Tagalog comparative choice: "Alin ang nauna: X o Y?"
            match = re.match(
                r"^\s*alin\s+ang\s+(?:nauna|sumunod)\s*:\s*(.+?)\s+(?:o|or)\s+(.+?)\s*$",
                value,
                flags=re.IGNORECASE,
            )
            if match:
                return [match.group(1).strip(), match.group(2).strip()]

            # English yes/no ordering, including "before or after".
            match = re.match(
                r"^\s*(?:was|is|did)\s+(.+?)\s+"
                r"(?:before\s+or\s+after|before|after|earlier\s+than|later\s+than)\s+"
                r"(.+?)\s*$",
                value,
                flags=re.IGNORECASE,
            )
            if match:
                return [match.group(1).strip(), match.group(2).strip()]

            # Comparative choice: "Which came later: A or B?"
            match = re.match(
                r"^\s*which\s+came\s+(?:later|first|earlier)\s*:\s*(.+?)\s+or\s+(.+?)\s*$",
                value,
                flags=re.IGNORECASE,
            )
            if match:
                return [match.group(1).strip(), match.group(2).strip()]

            # Generic comparison request with a later date/order instruction.
            match = re.match(
                r"^\s*compare\s+(.+?)\s+(?:with|to|and)\s+(.+?)(?:[.?!]|$)",
                value,
                flags=re.IGNORECASE,
            )
            if match:
                return [match.group(1).strip(), match.group(2).strip()]

            # "Put these two events in chronological order with dates: A and B"
            match = re.match(
                r"^\s*(?:put|place|arrange|order)\b.*?"
                r"\bchronological\s+order\b.*?:\s*(.+?)\s+(?:and|or|at|o)\s+(.+?)(?:[.?!]|$)",
                value,
                flags=re.IGNORECASE,
            )
            if match:
                return [match.group(1).strip(), match.group(2).strip()]

            parts = re.split(
                r"\s+\bkaysa\b\s+|\s+\b(?:before\s+or\s+after|before|after|earlier\s+than|later\s+than)\b\s+",
                value,
                maxsplit=1,
                flags=re.IGNORECASE,
            )
            return [part.strip() for part in parts] if len(parts) == 2 else []

        facets = split_facets(clean_question)
        if len(facets) != 2:
            return ""

        text = str(context)
        month_names = (
            r"January|February|March|April|May|June|July|August|"
            r"September|October|November|December"
        )
        full_date_pattern = re.compile(
            rf"\b(?:"
            rf"(?P<month_first>{month_names})\s+(?P<day_first>\d{{1,2}}),?\s+(?P<year_first>(?:18|19|20)\d{{2}})"
            rf"|"
            rf"(?P<day_second>\d{{1,2}})\s+(?P<month_second>{month_names})\s+(?P<year_second>(?:18|19|20)\d{{2}})"
            rf")\b",
            flags=re.IGNORECASE,
        )
        date_matches = list(full_date_pattern.finditer(text))
        dates = []
        for match_index, match in enumerate(date_matches):
            month_name = (
                match.group("month_first")
                or match.group("month_second")
            ).lower()
            day_value = int(
                match.group("day_first")
                or match.group("day_second")
            )
            year_value = int(
                match.group("year_first")
                or match.group("year_second")
            )
            key = (year_value, month_map[month_name], day_value)

            # Never let event words leak across one retrieved document into a
            # date that belongs to another document. Within a document, bind the
            # date to the smallest useful local unit: usually its sentence, or
            # nearby wrapped lines for PDF/table layouts such as:
            #   Treaty of Paris / Signed / December 10, 1898.
            previous_document = text.rfind("===== DOCUMENT", 0, match.start())
            next_document = text.find("===== DOCUMENT", match.end())
            document_start = 0 if previous_document < 0 else previous_document
            document_end = len(text) if next_document < 0 else next_document

            # The accepted Windows evidence exposed a subtle table/prose
            # boundary case: an "Effective <date>" field immediately before
            # a sentence about a different signing date could inherit the next
            # event's words. Build a tight date-local anchor window and stop it
            # at neighboring full dates. This keeps event/date binding local
            # without changing retrieval, thresholds, or reranking.
            local_start = document_start
            for previous_match in reversed(date_matches[:match_index]):
                if previous_match.start() < document_start:
                    break
                local_start = max(local_start, previous_match.end())
                break

            local_end = document_end
            for next_match in date_matches[match_index + 1:]:
                if next_match.start() >= document_end:
                    break
                local_end = min(local_end, next_match.start())
                break

            previous_sentence = max(
                text.rfind(".", document_start, match.start()),
                text.rfind("?", document_start, match.start()),
                text.rfind("!", document_start, match.start()),
            )
            sentence_start = document_start if previous_sentence < 0 else previous_sentence + 1
            next_sentence_candidates = [
                index for index in (
                    text.find(".", match.end(), document_end),
                    text.find("?", match.end(), document_end),
                    text.find("!", match.end(), document_end),
                )
                if index >= 0
            ]
            sentence_end = (
                min(next_sentence_candidates) + 1
                if next_sentence_candidates
                else document_end
            )
            sentence_window = text[sentence_start:sentence_end]

            # For wrapped PDF/table text, event labels generally precede the
            # value (for example ``Signed`` -> ``December 10, 1898``).  Walk
            # backward only, and never cross a neighboring full date.  This
            # is the critical guard that prevents ``Effective April 11, 1899``
            # from borrowing ``Treaty ... signed`` words that introduce the
            # *next* December 10 date.
            date_line_start = text.rfind("\n", document_start, match.start()) + 1
            date_line_end = text.find("\n", match.end(), document_end)
            if date_line_end < 0:
                date_line_end = document_end

            line_start = max(local_start, date_line_start)
            for _ in range(6):
                prior_break = text.rfind(
                    "\n",
                    local_start,
                    max(local_start, line_start - 1),
                )
                if prior_break < local_start:
                    line_start = local_start
                    break
                line_start = prior_break + 1
            line_window = text[line_start:min(date_line_end, local_end)]

            # Normal prose should bind within its own sentence.  Standalone
            # date/value rows and long PDF-wrapped pseudo-sentences instead
            # use the backward structural window above.  Choosing one local
            # unit (rather than a symmetric character radius) avoids letting
            # the words that introduce the next event contaminate this date.
            date_line = text[date_line_start:date_line_end]
            date_line_remainder = full_date_pattern.sub("", date_line, count=1)
            date_is_standalone = not date_line_remainder.strip(" \t\r,;:-()[]")
            sentence_is_compact = len(sentence_window.strip()) <= 420
            anchor_window = (
                sentence_window
                if sentence_is_compact and not date_is_standalone
                else line_window
            )

            useful_windows = [
                value for value in (sentence_window, line_window)
                if value and len(value.strip()) >= len(match.group(0)) + 8
            ]
            window = min(useful_windows, key=len) if useful_windows else anchor_window

            dates.append({
                "key": key,
                "label": re.sub(r"\s+", " ", match.group(0)).strip(),
                "month": month_map[month_name],
                "year": year_value,
                "window": canonical_event_text(window),
                "anchor_window": canonical_event_text(anchor_window),
            })

        if len({item["key"] for item in dates}) < 2:
            return ""

        stop = {
            "what", "when", "was", "were", "did", "is", "are", "the", "a", "an",
            "of", "to", "in", "on", "at", "noong", "ang", "ba", "pag", "sa", "ng",
            "mga", "than", "before", "after", "later", "earlier", "nauna", "sumunod",
            "kaysa", "date", "petsa", "which", "came", "or", "philippine", "philippines",
        }

        def facet_metadata(value: str):
            normalized = self._normalize_identity_match_text(value)
            raw_tokens = normalized.split()

            explicit_months = set()
            for token in raw_tokens:
                english_month = month_aliases.get(token, token)
                if english_month in month_map:
                    explicit_months.add(month_map[english_month])

            explicit_years = {
                int(year)
                for year in re.findall(r"\b(?:18|19|20)\d{2}\b", normalized)
            }

            canonical = canonical_event_text(value)
            tokens = []
            for token in canonical.split():
                if token in month_map or token in month_aliases:
                    continue
                if token in stop or token.isdigit() or len(token) < 3:
                    continue
                tokens.append(token)

            # Preserve order while deduplicating.  At least two anchors are
            # preferred when the facet provides them (e.g. sign+treaty,
            # declare+independence).
            unique_tokens = list(dict.fromkeys(tokens))
            return unique_tokens, explicit_months, explicit_years

        selected = []
        used_keys = set()
        for facet in facets:
            tokens, explicit_months, explicit_years = facet_metadata(facet)
            if not tokens:
                return ""

            ranked = []
            for item in dates:
                if item["key"] in used_keys:
                    continue
                if explicit_months and item["month"] not in explicit_months:
                    continue
                if explicit_years and item["year"] not in explicit_years:
                    continue

                # Score only anchors close to this exact date. The broader
                # sentence/line window remains available for diagnostics, but
                # it must not let a neighboring event borrow this date.
                anchor_tokens = set(item["anchor_window"].split())
                matched = [token for token in tokens if token in anchor_tokens]
                required_matches = 2 if len(tokens) >= 2 else 1
                if len(matched) < required_matches:
                    continue

                # Reward exact event-anchor coverage. Month/year constraints
                # are already hard filters above, not soft score bonuses. A
                # three-anchor event (for example action + object + qualifier)
                # naturally outranks a neighboring date that only shares two.
                score = (len(matched) * 10.0) + (len(matched) / max(1, len(tokens)))
                ranked.append((score, item, tuple(matched)))

            if not ranked:
                return ""

            ranked.sort(key=lambda row: row[0], reverse=True)
            best_score, best, _ = ranked[0]
            # If two different dates are equally plausible for the same event,
            # do not guess deterministically; let the normal grounded path deal
            # with it instead.
            if len(ranked) > 1 and ranked[1][0] == best_score and ranked[1][1]["key"] != best["key"]:
                return ""

            selected.append(best)
            used_keys.add(best["key"])

        first, second = selected
        if first["key"] == second["key"]:
            return ""

        earlier = first if first["key"] < second["key"] else second
        later = second if first["key"] < second["key"] else first
        first_is_earlier = first is earlier

        if re.match(
            r"^\s*(?:which\s+came\s+(?:later|first|earlier)|alin\s+ang\s+(?:nauna|sumunod))\b",
            clean_question,
            re.IGNORECASE,
        ):
            return f"{earlier['label']} came first; {later['label']} came later."

        if re.search(
            r"\b(?:chronological\s+order|compare)\b",
            clean_question,
            re.IGNORECASE,
        ):
            return f"{earlier['label']} came first; {later['label']} came later."

        if re.search(r"\b(?:nauna|before|earlier)\b", clean_question, re.IGNORECASE):
            if first_is_earlier:
                return f"Yes. {first['label']} came first; {second['label']} came later."
            return f"No. {second['label']} came first; {first['label']} came later."

        if re.search(r"\b(?:sumunod|after|later)\b", clean_question, re.IGNORECASE):
            if not first_is_earlier:
                return f"Yes. {first['label']} came later; {second['label']} came first."
            return f"No. {first['label']} came first; {second['label']} came later."

        return ""


    def _apply_structured_answer_focus(
        self,
        context: str,
        question: str,
        resolved_question: str,
        answer_focus: str,
        current_answer: str
    ):

        """Deterministically enforce exact structured answer intent."""

        reference = extract_structured_reference(
            resolved_question or question
        )

        if reference is None:
            return current_answer

        if reference.is_section_like:
            named_label = self._structured_detail_label(
                question,
                resolved_question,
            )
            if answer_focus.startswith("STRUCTURED DETAIL:") and named_label:
                preserve_layout = named_label == "Examples"
                example_display_label = "Examples"
                value = self._structured_labeled_block_from_context(
                    context,
                    named_label,
                    preserve_layout=preserve_layout,
                )
                if not value and named_label == "Examples":
                    example_display_label = "Example"
                    value = self._structured_labeled_block_from_context(
                        context,
                        "Example",
                        preserve_layout=True,
                    )
                if not value and named_label == "Exceptions":
                    value = self._structured_labeled_block_from_context(context, "Exception")
                if value:
                    if named_label == "Examples":
                        return self._format_structured_example_answer(
                            value,
                            display_label=example_display_label,
                        )
                    return f"{named_label}: {value}"
                return self._structured_missing_detail_message(
                    named_label,
                    reference.display_name,
                )

            if answer_focus.startswith("STRUCTURED SECTION TOPICS:"):
                topics = self._structured_section_topics_from_context(context, reference)
                return topics or current_answer

            if answer_focus.startswith("STRUCTURED SECTION OVERVIEW:"):
                title, intro = self._section_heading_and_intro_from_context(
                    context,
                    reference
                )

                if title:
                    base = f"{reference.display_name}: {title}"
                    if intro:
                        return f"{base}. {intro}"
                    return base

                return current_answer

            if answer_focus.startswith("STRUCTURED SECTION DETAIL:"):
                detail = self._structured_section_detail_from_context(
                    context=context,
                    reference=reference,
                    question=question,
                    resolved_question=resolved_question,
                )
                return detail or current_answer

            if answer_focus.startswith("STRUCTURED EXPLANATION:"):
                return self._structured_section_explanation_from_context(
                    context=context,
                    reference=reference,
                    current_answer=current_answer
                )
            return current_answer

        if reference.kind not in {"rule", "directive"}:
            return current_answer

        statement = self._structured_statement_from_context(
            context,
            reference
        )

        if not statement:
            return current_answer

        display_statement = self._clean_structured_statement_for_display(statement)

        display_name = (
            f"Directive {reference.identifier}"
            if reference.kind == "directive"
            and re.search(r"\bdirective\b", question or "", re.IGNORECASE)
            else reference.display_name
        )

        if answer_focus.startswith("STRUCTURED SIMPLE EXPLANATION:"):
            amplification = self._structured_labeled_block_from_context(
                context,
                "Amplification",
                stop_on_section_heading=False,
            )
            source_explanation = amplification or self._structured_rationale_from_context(context)
            source_explanation = re.sub(
                r"\s+",
                " ",
                str(source_explanation or ""),
            ).strip()
            source_explanation = re.sub(
                r"\s+([.,;:!?])",
                r"\1",
                source_explanation,
            )
            if source_explanation:
                source_explanation = re.split(
                    r"(?<=[.!?])\s+",
                    source_explanation,
                    maxsplit=1,
                )[0].strip()
            answer = f"**{display_name}:** {display_statement.rstrip('.')}."
            if source_explanation:
                answer += f"\n\n**In simple terms:** {source_explanation}"
            return answer

        if answer_focus.startswith("STRUCTURED EXPLANATION WITH CATEGORY:"):
            category = self._structured_labeled_block_from_context(
                context,
                "Category",
                stop_on_section_heading=False,
            )
            category = re.sub(r"\s+", " ", str(category or "")).strip()
            rationale = self._structured_rationale_from_context(context)
            rationale = self._first_meaningful_rationale_sentence(rationale)

            heading = f"**{display_name}" + (f" — {category}" if category else "") + "**"
            parts = [heading, display_statement.rstrip(".") + "."]
            if rationale:
                parts.append(f"**Rationale:** {rationale}")
            return "\n\n".join(parts)

        if answer_focus.startswith("STRUCTURED STATEMENT:"):
            return f"{display_name}: {display_statement}"

        if answer_focus.startswith("STRUCTURED DETAIL:"):
            clean_question = re.sub(
                r"\s+",
                " ",
                str(question or "").lower().strip(),
            )
            structured_question = f"{clean_question} {str(resolved_question or '').lower()}"

            label = self._structured_detail_label(
                question,
                resolved_question,
            )

            if label:
                preserve_layout = label == "Examples"
                example_display_label = "Examples"
                value = self._structured_labeled_block_from_context(
                    context,
                    label,
                    preserve_layout=preserve_layout,
                    stop_on_section_heading=False,
                )
                if not value and label == "Examples":
                    example_display_label = "Example"
                    value = self._structured_labeled_block_from_context(
                        context,
                        "Example",
                        preserve_layout=True,
                        stop_on_section_heading=False,
                    )
                if not value and label == "Exceptions":
                    value = self._structured_labeled_block_from_context(
                        context, "Exception", stop_on_section_heading=False
                    )
                if value:
                    if label == "Examples":
                        return self._format_structured_example_answer(
                            value,
                            display_label=example_display_label,
                        )

                    # A source Rationale can intentionally defer its real
                    # explanation to an exact cited Section (for example,
                    # "see Section X.Y.Z").  If that exact Section is already
                    # present in grounded context, answer from it instead of
                    # returning only the pointer sentence.  This applies to
                    # every explicit rationale phrasing, not one QA prompt.
                    if label == "Rationale":
                        expanded_rationale = self._rationale_cross_reference_answer(
                            context=context,
                            rationale=value,
                        )
                        if expanded_rationale:
                            return expanded_rationale

                    return f"{label}: {value}"

                # Label-preserving safety: an absent named block is not a
                # license to relabel rationale prose or inline phrases such as
                # "for example". The exact Rule/Directive was found, so say
                # explicitly that this named field is absent instead of using
                # the generic whole-KB fallback.
                return self._structured_missing_detail_message(
                    label,
                    display_name,
                )

            return current_answer

        rationale = self._structured_rationale_from_context(
            context
        )

        if answer_focus.startswith("STRUCTURED SUMMARY:"):
            summary = self._structured_summary_from_context(
                context=context,
                reference=reference,
                display_name=display_name,
            )
            return summary or f"{display_name}: {display_statement}"

        if answer_focus.startswith("STRUCTURED EXPLANATION WITH EXAMPLE:"):
            explanation = self._structured_explanation_from_context(
                context=context,
                reference=reference,
                display_name=display_name,
            ) or f"{display_name}: {display_statement}"
            example_label = "Examples"
            example = self._structured_labeled_block_from_context(
                context,
                "Examples",
                preserve_layout=True,
                stop_on_section_heading=False,
            )
            if not example:
                example_label = "Example"
                example = self._structured_labeled_block_from_context(
                    context,
                    "Example",
                    preserve_layout=True,
                    stop_on_section_heading=False,
                )
            if example:
                rendered_example = self._format_structured_example_answer(
                    example,
                    display_label=example_label,
                )
                if rendered_example:
                    explanation = explanation.rstrip() + "\n\n" + rendered_example
            else:
                explanation = (
                    explanation.rstrip()
                    + f"\n\n**Example:** No explicit Example section is provided "
                    f"in the source for {display_name}."
                )
            return explanation

        if answer_focus.startswith("STRUCTURED EXPLANATION:"):
            explanation = self._structured_explanation_from_context(
                context=context,
                reference=reference,
                display_name=display_name
            )

            if explanation:
                return explanation

            return f"{display_name}: {display_statement}"

        if answer_focus.startswith("REASON:") and rationale:
            expanded_reason = self._rationale_cross_reference_answer(
                context=context,
                rationale=rationale,
            )
            return expanded_reason or rationale

        return current_answer

    def _contains_prompt_leak(
        self,
        answer: str
    ):

        """
        Detect high-confidence prompt or instruction leakage.

        This intentionally checks prompt metadata and instruction
        signatures, not ordinary words that may also be valid facts.
        """

        if not answer:

            return False

        leak_patterns = [
            # Prompt section labels
            r"(?im)^\s*={3,}\s*$",
            r"(?im)^\s*core rules\s*$",
            r"(?im)^\s*primary document types\s*$",
            r"(?im)^\s*conversation history rules\s*$",
            r"(?im)^\s*short topic questions\s*$",
            r"(?im)^\s*list and multi-answer questions\s*$",
            r"(?im)^\s*specific questions\s*$",
            r"(?im)^\s*question focus priority\s*$",
            r"(?im)^\s*list and procedure format\s*$",
            r"(?im)^\s*missing or partial information\s*$",
            r"(?im)^\s*answer style\s*$",
            r"(?im)^\s*grounding rules\s*$",
            r"(?im)^\s*question focus\s*$",
            r"(?im)^\s*answer format\s*$",

            # Prompt data labels
            r"(?im)^\s*company knowledge\s*:",
            r"(?im)^\s*conversation history\s*:",
            r"(?im)^\s*user question\s*:",
            r"(?im)^\s*answer focus\s*:",
            r"(?im)^\s*required answer focus\s*:",
            r"(?im)^\s*required answer type\s*:",
            r"(?im)^\s*draft answer\s*:",
            r"(?im)^\s*search query\s*:",

            # Exact instruction signatures
            r"(?i)\byour only source of truth is\b",
            r"(?i)\buse only the company knowledge\b",
            r"(?i)\banswer only using the provided company knowledge\b",
            r"(?i)\bdo not use outside knowledge\b",
            r"(?i)\bdo not answer from memory\b",
            r"(?i)\bdo not mention context\b",
            r"(?i)\breturn only the final answer\b",
            r"(?i)\bif the answer is not found in the company knowledge\b",
            r"(?i)\byou are docubot\b",
            r"(?i)\byou are checking a draft answer\b",
            r"(?i)\byou are validating whether a draft answer\b",
            r"(?i)\brewrite the user's question into a clear standalone search query\b",

            # Verifier commentary must never reach the UI.
            r"(?i)\bthe draft answer\b",
            r"(?i)\bdraft answer does not match\b",
            r"(?i)\brequired answer focus\b",
            r"(?i)\bresolved question or target\b",
            r"(?i)\bthe correct answer is\s*:",
            r"(?i)\bdoes not match the required answer\b",
            r"(?i)\bchecking a draft answer\b",
        ]

        for pattern in leak_patterns:

            if re.search(
                pattern,
                answer
            ):

                return True

        return False

    @staticmethod
    def _presentation_sentences(answer: str):

        if not answer:
            return []

        clean = re.sub(r"\s+", " ", str(answer)).strip()
        if not clean:
            return []

        sentences = re.split(
            r"(?<=[.!?])\s+(?=(?:[A-ZÀ-ÖØ-Þ0-9\"“]))",
            clean,
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def _format_answer_presentation(
        self,
        answer: str,
        question: str,
        answer_focus: str,
    ):

        """Apply deterministic presentation only; never add factual content.

        Multi-point explanations and compound answers are rendered as Markdown
        bullets so formatting does not depend on a model choosing bullets.
        Existing lists/code are preserved. Simple one-fact answers stay concise.
        """

        if not answer:
            return answer

        clean = str(answer).strip()
        fallback = NO_RESULT_MESSAGE.strip()

        if (
            not clean
            or clean.lower() == fallback.lower()
            or "```" in clean
        ):
            return clean

        # Identity lead paragraphs extracted from PDF infoboxes can carry
        # repeated headings, IPA pronunciation blocks, and footnote markers.
        # Remove only that presentation noise; keep the supported predicate.
        if (
            str(answer_focus or "").startswith("IDENTITY OR OVERVIEW:")
            and re.search(
                r"(?i)\b(?:tagalog|spanish)\s*:|"
                r"[ɐ-ʯ]",
                clean,
            )
        ):
            clean = self._clean_grounded_identity_sentence(
                clean,
                question,
                "",
            )

        lines = clean.splitlines()
        non_empty = [line.strip() for line in lines if line.strip()]

        list_lines = [
            line
            for line in non_empty
            if re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", line)
        ]

        focus = str(answer_focus or "")

        # For explicit list intent, even a one-item deterministic list is a
        # real list. Preserve its heading, scope note, and bullet structure.
        # The old generic presenter flattened one-item structured lists such as
        # compiler-option guidance into malformed bullets.
        if focus.startswith("LIST:") and list_lines:
            return clean

        # A single numbered/bulleted response outside list intent is not a list
        # and looks awkward in the UI. Keep only its factual text.
        if len(non_empty) == 1 and len(list_lines) == 1:
            return re.sub(
                r"^(?:[-*+]\s+|\d+[.)]\s+)",
                "",
                non_empty[0],
            ).strip()

        # Respect already-useful Markdown lists produced by deterministic
        # structured fallback or the model.
        if len(list_lines) >= 2:
            return clean

        # Preserve explicit source-shaped structured blocks. v6.4.29 Windows
        # evidence showed that the generic explanation presenter flattened
        # Category / Analysis / Applies to / Rationale into one bullet even
        # though the deterministic exact-rule formatter had produced separate
        # source labels correctly.
        structured_labels = sum(
            1
            for line in non_empty
            if re.match(
                r"^(?:Category|Analysis|Applies\s+to|Amplification|Rationale|"
                r"Example|Examples|Exception|Exceptions|See\s+also|Why\s+it\s+applies)\s*:?$",
                line,
                flags=re.IGNORECASE,
            )
        )
        if structured_labels >= 2:
            return clean

        is_explanation = focus.startswith(
            ("STRUCTURED EXPLANATION:", "GROUNDED EXPLANATION:")
        )
        is_compound = focus.startswith("COMPOUND:")
        is_list = focus.startswith("LIST:")

        if not (is_explanation or is_compound or is_list):
            return clean

        if is_list and len(non_empty) >= 2 and not list_lines:
            return "\n".join(f"- {line}" for line in non_empty)

        if is_list and ":" in clean and "," in clean:
            lead, values = clean.split(":", 1)
            parts = [
                re.sub(r"^(?:and|at)\s+", "", part.strip(" .,"), flags=re.IGNORECASE)
                for part in re.split(r"\s*,\s*|\s+(?:and|at)\s+", values)
                if part.strip(" .,")
            ]
            if 2 <= len(parts) <= 12 and all(len(part.split()) <= 12 for part in parts):
                return "\n".join(f"- {part}" for part in parts)

        sentences = self._presentation_sentences(clean)

        # For list intent, also support a semicolon-delimited single line.
        if len(sentences) < 2 and is_list and clean.count(";") >= 1:
            parts = [
                part.strip(" ,;")
                for part in clean.split(";")
                if part.strip(" ,;")
            ]
            if len(parts) >= 2:
                return "\n".join(f"- {part}" for part in parts)

        if len(sentences) < 2:
            return clean

        if is_compound or is_list:
            formatted = "\n".join(
                f"- {sentence}"
                for sentence in sentences
            )
        else:
            # Structured/general explanations usually read best with the
            # opening statement retained as the lead, followed by one bullet
            # per distinct supported point. For a two-sentence explanation,
            # both sentences are bullets to avoid a heading plus one-item list.
            if len(sentences) >= 3:
                formatted = (
                    sentences[0]
                    + "\n\n"
                    + "\n".join(
                        f"- {sentence}"
                        for sentence in sentences[1:]
                    )
                )
            else:
                formatted = "\n".join(
                    f"- {sentence}"
                    for sentence in sentences
                )

        evidence_logger.record_event(
            event_name="ANSWER PRESENTATION",
            status="LIST FORMAT APPLIED",
            details={
                "focus": focus.split(":", 1)[0] if ":" in focus else focus,
                "items": len(sentences),
            },
        )

        return formatted.strip()

    def _apply_output_safety_gate(
        self,
        answer: str,
        question: str = ""
    ):

        """
        Final deterministic guard before anything is returned to the UI.

        If prompt leakage is detected, discard the whole response
        instead of exposing partial prompt content.
        """

        fallback = NO_RESULT_MESSAGE.strip()

        # If a verifier ignored its output contract, recover the explicit
        # final ANSWER section before checking for prompt leakage. This keeps
        # internal verifier labels from causing a false fallback while still
        # rejecting genuine prompt/instruction leakage.
        cleaned = self._extract_verifier_answer(
            answer,
            ""
        )

        cleaned = self._strip_output_wrappers(
            cleaned
        )

        cleaned = self._postprocess_answer(
            cleaned,
            question
        )

        if not cleaned:

            return fallback

        if self._contains_prompt_leak(
            cleaned
        ):

            if DEBUG_MODE:

                print(
                    "\n[OUTPUT SAFETY GATE] "
                    "Prompt leakage detected. "
                    "Response replaced with fallback."
                )

            return fallback

        return cleaned

    @staticmethod
    def _has_substantive_answer_content(answer: str):
        """Reject punctuation/Markdown-only model artifacts such as ``>``."""
        if not answer:
            return False
        tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", str(answer))
        return len(tokens) >= 1

    def _recover_compact_source_fact(
        self,
        question: str,
        results,
    ):
        """Recover one direct source sentence after a non-substantive model reply.

        This is a last-resort grounding safeguard, not a general answer path.
        It activates only when generation produced no usable words and only
        returns a sentence/line copied from accepted company context with
        strong lexical overlap plus an explicit requested value signal.
        """
        if not question or not results:
            return ""

        normalized_question = self._normalize_identity_match_text(question)
        stop = {
            "what", "which", "who", "when", "where", "why", "how",
            "the", "a", "an", "of", "to", "for", "in", "on", "at",
            "and", "or", "with", "from", "is", "are", "was", "were",
            "do", "does", "did", "give", "show", "state", "return",
            "ang", "ng", "mga", "sa", "ay", "at", "ano", "anong",
            "sino", "alin", "kailan", "saan", "gaano", "dapat", "ayon",
        }

        def stem(token):
            token = token.lower()
            if token.endswith("ies") and len(token) > 5:
                return token[:-3] + "y"
            if token.endswith("ing") and len(token) > 6:
                return token[:-3]
            if token.endswith("ed") and len(token) > 5:
                return token[:-2]
            if token.endswith("es") and len(token) > 5:
                return token[:-2]
            if token.endswith("s") and len(token) > 4:
                return token[:-1]
            if token.endswith("e") and len(token) > 4:
                return token[:-1]
            return token

        query_tokens = []
        seen = set()
        for token in re.findall(r"[a-z0-9]+", normalized_question):
            if len(token) <= 2 or token in stop:
                continue
            key = stem(token)
            if key not in seen:
                seen.add(key)
                query_tokens.append(key)

        if len(query_tokens) < 2:
            return ""

        value_intent = bool(re.search(
            r"\b(?:how\s+many|how\s+much|how\s+often|gaano\s+kadalas|"
            r"interval|frequency|amount|rate|when|date|kailan|petsa)\b",
            normalized_question,
            flags=re.IGNORECASE,
        ))
        value_pattern = re.compile(
            r"\b\d+(?:\.\d+)?\b|"
            r"\b(?:daily|weekly|monthly|quarterly|yearly|annually|annual|"
            r"every\s+\d+\s+(?:day|days|week|weeks|month|months|year|years))\b|"
            r"\b(?:january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\b",
            flags=re.IGNORECASE,
        )

        candidates = []
        for item in list(results)[:3]:
            text = str(item.get("text", "") or "")
            if not text:
                continue
            segments = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                segments.extend(
                    segment.strip()
                    for segment in re.split(r"(?<=[.!?])\s+", line)
                    if segment.strip()
                )

            for segment in segments:
                normalized_segment = self._normalize_identity_match_text(segment)
                segment_tokens = {
                    stem(token)
                    for token in re.findall(r"[a-z0-9]+", normalized_segment)
                    if len(token) > 2
                }
                matched = [token for token in query_tokens if token in segment_tokens]
                if len(matched) < 2:
                    continue
                coverage = len(matched) / len(query_tokens)
                if coverage < 0.40:
                    continue
                if value_intent and not value_pattern.search(segment):
                    continue

                score = coverage + (0.25 if value_pattern.search(segment) else 0.0)
                candidates.append((score, segment))

        if not candidates:
            return ""

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1].strip()

    def _postprocess_answer(
        self,
        answer: str,
        question: str = ""
    ):

        """
        Clean the LLM response before returning it.
        Always return the fallback when cleanup produces
        an empty answer.
        """

        fallback = NO_RESULT_MESSAGE.strip()

        # Protect against None or empty model output.
        if not answer:

            return fallback

        answer = answer.strip()

        if not answer:

            return fallback

        # Exact fallback response.
        if answer.lower() == fallback.lower():

            return fallback

        # Remove accidental fallback appended to another response.
        answer = re.sub(
            re.escape(fallback),
            "",
            answer,
            flags=re.IGNORECASE
        ).strip()

        # If removing the fallback left nothing,
        # restore the proper fallback message.
        if not answer:

            return fallback

        unwanted_phrases = [
            "I'm DocuBot, a company knowledge assistant.",
            "I’m DocuBot, a company knowledge assistant.",
            "Based on the provided company knowledge,",
            "According to the provided company knowledge,",
            "I'll provide an answer based on the provided company knowledge.",
            "I will provide an answer based on the provided company knowledge."
        ]

        for phrase in unwanted_phrases:

            answer = answer.replace(
                phrase,
                ""
            )

        answer = self._remove_repeated_question(
            answer,
            question
        )

        # Convert bullet symbols into Markdown bullets.
        answer = re.sub(
            r"(?m)^\s*[•●▪]\s+",
            "- ",
            answer
        )

        answer = re.sub(
            r"\s+[•●▪]\s+",
            "\n- ",
            answer
        )

        while "\n\n\n" in answer:

            answer = answer.replace(
                "\n\n\n",
                "\n\n"
            )

        answer = answer.strip()

        # Final safety check.
        if not answer:

            return fallback

        return answer

    def _remove_repeated_question(
        self,
        answer: str,
        question: str
    ):

        """
        Remove exact or partial restatements of the user's question
        from the beginning of the answer.

        This works across languages because it compares Unicode words
        instead of relying on English or Tagalog keywords.
        """

        if not answer or not question:

            return answer

        def normalize_text(
            text: str
        ):

            text = str(
                text
                or ""
            ).strip().lower()

            text = re.sub(
                r"[^\w\s]",
                " ",
                text,
                flags=re.UNICODE
            )

            text = re.sub(
                r"\s+",
                " ",
                text
            ).strip()

            return text

        def get_tokens(
            text: str
        ):

            return [
                token
                for token in normalize_text(
                    text
                ).split()
                if token
            ]

        normalized_question = normalize_text(
            question
        )

        question_tokens = set(
            get_tokens(
                question
            )
        )

        if not normalized_question:

            return answer

        lines = answer.splitlines()

        cleaned_lines = []

        # Only inspect the first few non-empty lines.
        # This prevents legitimate later content from being changed.
        inspected_non_empty_lines = 0

        for line in lines:

            stripped_line = line.strip()

            if not stripped_line:

                cleaned_lines.append(
                    line
                )

                continue

            inspected_non_empty_lines += 1

            if inspected_non_empty_lines > 4:

                cleaned_lines.append(
                    line
                )

                continue

            # Preserve the original Markdown marker when a
            # repeated question prefix has a direct answer after ":".
            marker_match = re.match(
                r"^(\s*(?:[-*+]|\d+[.)])\s+)(.*)$",
                line
            )

            if marker_match:

                marker = marker_match.group(1)
                content = marker_match.group(2).strip()

            else:

                marker = ""
                content = stripped_line

            normalized_content = normalize_text(
                content
            )

            # Exact repeated question.
            if (
                normalized_content
                == normalized_question
            ):

                continue

            # Detect:
            # "<question wording>: <direct answer>"
            # or:
            # "<question wording>:"
            if ":" in content:

                prefix, suffix = content.split(
                    ":",
                    1
                )

                prefix_tokens = get_tokens(
                    prefix
                )

                prefix_token_set = set(
                    prefix_tokens
                )

                overlap_ratio = 0.0

                if prefix_token_set:

                    overlap_ratio = (
                        len(
                            prefix_token_set.intersection(
                                question_tokens
                            )
                        )
                        / len(
                            prefix_token_set
                        )
                    )

                # Conservative rule:
                # - at least four words;
                # - nearly all prefix words came from the question.
                is_question_restatement = (
                    len(prefix_tokens) >= 4
                    and overlap_ratio >= 0.80
                )

                if is_question_restatement:

                    direct_answer = suffix.strip()

                    # Heading-like repetition with no answer.
                    if not direct_answer:

                        continue

                    # Keep only the direct answer after the colon.
                    cleaned_lines.append(
                        f"{marker}{direct_answer}"
                        if marker
                        else direct_answer
                    )

                    continue

            cleaned_lines.append(
                line
            )

        cleaned_answer = "\n".join(
            cleaned_lines
        ).strip()

        return (
            cleaned_answer
            or answer.strip()
        )

    def _is_multi_answer_question(
        self,
        question: str
    ):

        """
        Detect questions that require multiple items.

        This is generic:
        - people
        - requirements
        - rules
        - steps
        - options
        - parameters
        - examples
        """

        if not question:

            return False

        clean = question.lower().strip()

        indicators = [
            "who are",
            "what are",
            "list",
            "enumerate",
            "name the",
            "give me the list",
            "examples",
            "types",
            "categories",
            "requirements",
            "rules",
            "steps",
            "procedure",
            "procedures",
            "process",
            "options",
            "parameters",
            "items",
        ]

        if any(
            indicator in clean
            for indicator in indicators
        ):

            return True

        # Multi-value requests are also list intent even when the grammar uses
        # one interrogative over several named fields rather than "what are".
        if re.search(r"^how\s+(?:many|much)\b.*\b(?:and|at)\b", clean):
            return True

        if re.search(r"^(?:give|provide|state|show)\b", clean) and clean.count(",") >= 2:
            return True

        return self._looks_like_plural_list_question(
            question
        )

    def _verify_multi_answer(
        self,
        context: str,
        question: str,
        draft_answer: str,
        resolved_question: str = "",
        llm_client=None,
    ):

        """
        Verify list / multi-answer output.

        Purpose:
        If the first LLM answer missed items that are already
        in the retrieved context, ask the LLM to correct it.

        This does NOT use outside knowledge.
        This does NOT use reranker.
        """

        fallback = NO_RESULT_MESSAGE.strip()

        if not context or not draft_answer:

            return draft_answer

        if draft_answer.strip() == fallback:

            return draft_answer

        verify_prompt = f"""
You are checking a draft answer for completeness.

Use ONLY the COMPANY KNOWLEDGE below.

Do not use outside knowledge.
Do not guess.
Do not invent missing items.
Do not mention documents, sources, context, or verification.

COMPANY KNOWLEDGE:
{context}

USER QUESTION:
{question}

RESOLVED QUESTION OR TARGET:
{resolved_question or question}

DRAFT ANSWER:
{draft_answer}

Task:
1. Independently scan every COMPANY KNOWLEDGE section from first to last.
2. Match items to the exact relationship or category expressed by the
   RESOLVED QUESTION OR TARGET.
3. Check headings, paragraphs, continuation sections, comma-separated lists,
   semicolon-separated lists, and table-like text.
4. Do not stop after the first paragraph or first list.
5. Include every explicitly supported relevant item.
6. Merge duplicate references to the same item.
7. Compare the complete set with the DRAFT ANSWER.
8. If the DRAFT ANSWER missed relevant items, return a corrected final answer.
9. Use one Markdown bullet per item when multiple items are present.
10. Do not invent missing items.
11. Return only the final answer.
"""

        try:

            verifier = llm_client or self.llm
            verified_answer = self._generate_with_latency(
                verifier,
                verify_prompt,
                purpose="multi_answer_verification",
            )

            verified_answer = self._postprocess_answer(
                verified_answer,
                question
            )

            if not verified_answer:

                return draft_answer

            return verified_answer

        except Exception:

            return draft_answer

    def _compound_verifier_repair_is_safe(
        self,
        context: str,
        draft_answer: str,
        verified_answer: str,
    ):

        """Reject verifier outputs that expand beyond a narrow repair.

        Compound verification is a validation/repair step, not a second answer
        generator. A candidate may fix a missing facet, but it must remain
        compact and every newly introduced content unit must have strong
        lexical support in the already accepted company context.
        """

        if not verified_answer:
            return False

        if verified_answer.strip() == draft_answer.strip():
            return True

        def tokens(text):
            normalized = self._normalize_identity_match_text(text)
            stopwords = {
                "a", "an", "the", "and", "or", "of", "to", "in",
                "on", "for", "from", "with", "by", "as", "at",
                "is", "was", "are", "were", "be", "been", "being",
                "it", "its", "this", "that", "these", "those",
                "he", "she", "they", "them", "his", "her", "their",
                "can", "could", "should", "would", "will", "may",
                "also", "only", "all", "each", "every",
            }
            return [
                token for token in normalized.split()
                if len(token) >= 2 and token not in stopwords
            ]

        draft_words = tokens(draft_answer)
        verified_words = tokens(verified_answer)
        max_extra = max(18, int(len(draft_words) * 0.35))

        if len(verified_words) > len(draft_words) + max_extra:
            return False

        context_token_set = set(tokens(context))

        def units(text):
            raw_units = re.split(
                r"(?:\n+|(?<=[.!?])\s+)",
                str(text).strip()
            )
            return [
                re.sub(r"^\s*(?:[-*+] |\d+[.)]\s*)", "", unit).strip()
                for unit in raw_units
                if unit.strip()
            ]

        draft_unit_tokens = [set(tokens(unit)) for unit in units(draft_answer)]

        for unit in units(verified_answer):
            unit_tokens = set(tokens(unit))

            if len(unit_tokens) < 2:
                continue

            # Existing/rephrased draft content is not a new factual addition.
            if any(
                base and len(unit_tokens.intersection(base)) / max(1, len(unit_tokens)) >= 0.70
                for base in draft_unit_tokens
            ):
                continue

            support_ratio = (
                len(unit_tokens.intersection(context_token_set))
                / max(1, len(unit_tokens))
            )

            if support_ratio < 0.60:
                return False

        return True


    def _verify_compound_answer(
        self,
        context: str,
        question: str,
        draft_answer: str,
        resolved_question: str = "",
        llm_client=None,
    ):

        """
        Ensure that every independently requested clause is answered.

        This verifier uses only retrieved company knowledge.
        """

        fallback = NO_RESULT_MESSAGE.strip()

        if (
            not context
            or not draft_answer
            or draft_answer.strip().lower()
            == fallback.lower()
        ):

            return draft_answer

        if self._compound_draft_can_skip_verifier(
            question=question,
            draft_answer=draft_answer,
        ):
            evidence_logger.record_event(
                event_name="LATENCY OPTIMIZATION",
                status="COMPOUND VERIFIER SKIPPED",
                details={
                    "reason": (
                        "Every explicit compound facet passed the conservative "
                        "deterministic coverage gate."
                    )
                },
            )
            return draft_answer

        verify_prompt = f"""
Validate and minimally repair the draft answer to a multi-part question.

Use ONLY the COMPANY KNOWLEDGE below.
Do not use outside knowledge.
Do not guess.
Do not invent facts.
Do not add background, examples, implications, interpretations, or extra details
that are not required to repair a missing requested part.
Do not mention prompts, verification, context, documents, or sources.

COMPANY KNOWLEDGE:
{context}

USER QUESTION:
{question}

RESOLVED QUESTION OR TARGET:
{resolved_question or question}

DRAFT ANSWER:
{draft_answer}

Instructions:
1. Identify every independently requested part of the USER QUESTION.
2. If the DRAFT ANSWER already covers every requested part, return it unchanged.
3. If one requested part is missing, repair only that missing part and preserve
   the rest of the DRAFT ANSWER as closely as possible.
4. Do not add optional supporting details merely because they appear in COMPANY
   KNOWLEDGE. This step is validation/repair only, not answer expansion.
5. Use only facts explicitly supported by COMPANY KNOWLEDGE.
6. Keep the repaired answer concise and in the original requested order.
7. If one requested part is not supported, state the exact fallback only for
   that unsupported part; do not invent an answer.
8. Return only the final user-facing answer.
"""

        try:

            verifier = llm_client or self.llm
            verified_answer = self._generate_with_latency(
                verifier,
                verify_prompt,
                purpose="compound_answer_verification",
            )

            verified_answer = self._postprocess_answer(
                verified_answer,
                question
            )

            if not verified_answer:

                return draft_answer

            if not self._compound_verifier_repair_is_safe(
                context=context,
                draft_answer=draft_answer,
                verified_answer=verified_answer,
            ):
                evidence_logger.record_event(
                    event_name="GROUNDING VERIFIER GUARD",
                    status="EXPANSION REJECTED",
                    details={
                        "reason": (
                            "Compound verifier output exceeded the narrow "
                            "repair boundary or introduced a weakly grounded "
                            "new content unit. Original grounded draft kept."
                        )
                    },
                )
                return draft_answer

            return verified_answer

        except Exception:

            return draft_answer


    def _extract_topic_from_results(
        self,
        results
    ):

        """
        Extract a canonical topic from the top source file.

        Examples:
            José Rizal - Wikipedia.pdf
            -> José Rizal

            Emilio Aguinaldo - Wikipedia.pdf
            -> Emilio Aguinaldo

            Employee Leave Policy.docx
            -> Employee Leave Policy
        """

        if not results:

            return None

        metadata = (
            results[0]
            .get(
                "metadata",
                {}
            )
        )

        file_name = decode_unicode_markers(
            metadata.get(
                "file_name",
                ""
            )
        ).strip()

        if not file_name:

            return None

        # Remove file extension
        topic = Path(file_name).stem

        # Remove common source suffixes
        topic = re.sub(
            r"\s*-\s*Wikipedia$",
            "",
            topic,
            flags=re.IGNORECASE
        )

        topic = re.sub(
            r"[\s_-]*FromInternet$",
            "",
            topic,
            flags=re.IGNORECASE
        )

        topic = re.sub(
            r"\s+",
            " ",
            topic
        ).strip()

        if not topic:

            return None

        return topic

    def _clean_retrieval_query(
        self,
        query: str
    ):

        """
        Clean a generated retrieval query without changing meaning.
        """

        if not query:

            return ""

        query = str(
            query
        ).strip()

        query = re.sub(
            r"(?im)^\s*(?:english\s+search\s+query|search\s+query|query)\s*:\s*",
            "",
            query,
            count=1
        )

        query = query.strip(
            " `\"'"
        )

        query = re.sub(
            r"\s+",
            " ",
            query
        ).strip()

        return query[
            :MULTILINGUAL_QUERY_MAX_CHARS
        ].strip()

    @staticmethod
    def _deterministic_multilingual_relation_query(
        question: str,
    ):

        """Build a cheap canonical relation query for common Tagalog cues.

        This is not a document/domain translation table. It only normalizes
        generic question/relation words while preserving the user's content
        terms. The original-language query is still searched in parallel.
        """

        if not question:
            return ""

        clean = re.sub(
            r"\s+",
            " ",
            str(question).strip()
        )

        # High-confidence relation words only. Unknown content words, names,
        # identifiers, numbers, and technical terms remain untouched.
        replacements = (
            # Phrase-level relations first so word-level substitutions cannot
            # fragment the intended meaning.
            (r"\bnais\s+makamit\b", "goal"),
            (r"\bgustong?\s+makamit\b", "goal"),
            (r"\bmagbigay\s+ng\s+pahintulot\b", "approve"),
            (r"\bayon\s+sa\b", ""),
            (r"\bkailan\b", "when"),
            (r"\bpetsa\b", "date"),
            (r"\bnilagdaan\b", "signed"),
            (r"\bpagpirma\b", "signing"),
            (r"\bitinatag\b", "founded"),
            (r"\bnaitatag\b", "founded"),
            (r"\bpagkakatatag\b", "founding"),
            (r"\bnagtatag\b", "founder"),
            (r"\bnabuo\b", "formed"),
            (r"\bnagsimula\b", "started"),
            (r"\bnatapos\b", "ended"),
            (r"\binilabas\b", "issued"),
            (r"\bsino\b", "who"),
            (r"\bano\b", "what"),
            (r"\banong\b", "what"),
            (r"\bbakit\b", "why"),
            (r"\bpaano\b", "how"),
            (r"\bsaan\b", "where"),
            (r"\balin\b", "which"),
            (r"\bilang\b", "how many"),
            (r"\bgaano\b", "how much"),
            (r"\baraw\b", "days"),
            (r"\btaunang\b", "annual"),
            (r"\bpangunahing\b", "main"),
            (r"\blayunin\b", "purpose"),
            (r"\badhikain\b", "purpose"),
            (r"\bmithiin\b", "goal"),
            (r"\bhangarin\b", "goal"),
            (r"\bmahalaga\b", "important"),
            (r"\bkahalagahan\b", "importance"),
            (r"\bposisyon\b", "position"),
            (r"\bpamahalaan\b", "government"),
            (r"\bhinawakan\b", "held"),
            (r"\bteritoryo(?:ng)?\b", "territories"),
            (r"\bbinitiwan\b", "relinquished"),
            (r"\bisinuko\b", "ceded"),
            (r"\bdeklarasyon\b", "declaration"),
            (r"\bkalayaan\b", "independence"),
            (r"\bhunyo\b", "June"),
            (r"\bnauna\b", "before"),
            (r"\bpahintulot\b", "approval"),
            (r"\bmaaprubahan\b", "approved"),
            (r"\bdadaan\b", "requires"),
            (r"\b(?:ibinibigay|binibigay|ibinigay)\b", "provided"),
            (r"\b(?:ibigay|magbigay)\b", "provide"),
            (r"\b(?:nag[- ]?aapruba|nag[- ]?aapprove|umaapruba|umaapprove)\b", "approves"),
            (r"\b(?:inaaprubahan|inaprubahan)\b", "approved"),
            (r"\bapruba(?:han)?\b", "approve"),
            (r"\bpag-apruba\b", "approval"),
            (r"\bkailangan(?:g)?\b", "required"),
        )

        translated = clean
        replacement_count = 0

        for pattern, replacement in replacements:
            translated, count = re.subn(
                pattern,
                replacement,
                translated,
                flags=re.IGNORECASE
            )
            replacement_count += count

        if replacement_count == 0:
            return ""

        # Remove only standalone Tagalog grammar particles from this
        # supplemental canonical query. The untouched original query remains
        # part of retrieval, so exact names/titles are never lost globally.
        translated = re.sub(
            r"\b(?:ang|ng|mga|ay|ba|si|ni|kay|na|nang|sa)\b",
            " ",
            translated,
            flags=re.IGNORECASE
        )
        translated = re.sub(
            r"[?!.]+$",
            "",
            translated
        )
        translated = re.sub(
            r"\s+",
            " ",
            translated
        ).strip()

        return translated


    def _is_clearly_english_query(
        self,
        question: str
    ):

        """
        Detect questions that are clearly English.

        Ambiguous or non-English text is allowed to use the
        multilingual English-query adapter.
        """

        if not question:

            return True

        clean = re.sub(
            r"\s+",
            " ",
            question.lower().strip()
        )

        # Any letter outside the Latin script is treated as
        # potentially non-English.
        for character in clean:

            if not character.isalpha():

                continue

            try:

                import unicodedata

                character_name = unicodedata.name(
                    character
                )

            except Exception:

                return False

            if (
                "LATIN" not in character_name
                and "COMBINING" not in character_name
            ):

                return False

        english_starters = (
            "who",
            "what",
            "when",
            "where",
            "why",
            "how",
            "which",
            "is",
            "are",
            "was",
            "were",
            "do",
            "does",
            "did",
            "can",
            "could",
            "would",
            "should",
            "will",
            "tell",
            "explain",
            "describe",
            "define",
            "list",
            "name",
            "give",
            "show",
            "please",
        )

        first_word_match = re.match(
            r"^[a-z]+",
            clean
        )

        first_word = (
            first_word_match.group(0)
            if first_word_match
            else ""
        )

        # Strong Tagalog query cues must win before generic English function
        # words. A proper title such as "Treaty of Paris" contains "of"
        # but that does not make "Kailan nilagdaan ..." an English query.
        tagalog_query_cues = {
            "ano", "anong", "sino", "kailan", "saan", "bakit",
            "paano", "alin", "ilang", "gaano", "may",
        }

        # English modal "May" must not be confused with the Tagalog
        # existential/possessive word "may".  Require an English determiner
        # immediately after it so Tagalog questions such as "May parameter ..."
        # remain eligible for multilingual normalization while sentences such as
        # "May a goto jump ...?" stay on the zero-translation English path.
        if first_word == "may" and re.match(
            r"^may\s+(?:a|an|the|this|that|these|those)\b",
            clean,
            flags=re.IGNORECASE,
        ):
            return True

        if first_word in tagalog_query_cues:
            return False

        # Mixed-language questions can start with an English interrogative but
        # still contain a strong Tagalog relation/source phrase. Treat those as
        # multilingual so the deterministic canonicalizer and normal retry path
        # remain available. These are grammar cues, not document/domain terms.
        if re.search(
            r"\b(?:ayon\s+sa|tungkol\s+sa|tungkol\s+kay|para\s+sa|mula\s+sa)\b",
            clean,
            flags=re.IGNORECASE,
        ):
            return False

        if first_word in english_starters:
            return True

        words_in_order = re.findall(
            r"[a-z]+",
            clean
        )
        tagalog_function_words = {
            "ang", "ng", "mga", "ay", "si", "ni", "kay", "para",
            "mula", "kung", "dahil", "ito", "iyan", "iyon", "siya",
            "niya", "nito", "din", "rin",
        }

        if len(
            set(words_in_order).intersection(tagalog_function_words)
        ) >= 2:
            return False

        english_function_words = {
            "the",
            "of",
            "and",
            "or",
            "for",
            "from",
            "with",
            "about",
            "into",
            "during",
            "before",
            "after",
            "through",
            "between",
            "against",
            "this",
            "that",
            "these",
            "those",
            "his",
            "her",
            "their",
            "its",
        }

        words = set(
            re.findall(
                r"[a-z]+",
                clean
            )
        )

        if words.intersection(english_function_words):
            return True

        # Technical/document queries are often noun phrases rather than full
        # sentences (for example "ISO C portability issue references"). They
        # may contain no English function word and previously paid for an
        # unnecessary LLM "translation" call.  When the text is Latin-script,
        # contains no strong Tagalog cue, and includes ordinary English
        # document/technical vocabulary, treat it as English directly.
        english_technical_terms = {
            "reference", "references", "issue", "issues", "standard", "standards",
            "section", "sections", "rule", "rules", "directive", "directives",
            "guideline", "guidelines", "requirement", "requirements",
            "policy", "policies", "procedure", "procedures", "process", "processes",
            "system", "systems", "document", "documents", "manual", "guide",
            "code", "compiler", "toolchain", "option", "options", "flag", "flags",
            "statement", "statements", "configuration", "setup", "overview",
            "summary", "category", "classification", "example", "examples",
            "rationale", "purpose", "status", "eligibility", "approval",
            "portability", "implementation", "undefined", "unspecified",
        }

        if len(words) >= 2 and words.intersection(english_technical_terms):
            return True

        return False

    def _build_english_retrieval_query(
        self,
        question: str,
        history: str = "",
        current_topic: str = ""
    ):

        """
        Build a standalone English retrieval query for a
        non-English or ambiguous-language question.

        This method does not answer the question.
        """

        if (
            not ENABLE_MULTILINGUAL_RETRIEVAL
            or not question
        ):

            return ""

        cache_key = (
            str(
                current_topic
                or ""
            ).strip().lower(),
            str(
                history
                or ""
            ).strip().lower(),
            str(
                question
                or ""
            ).strip().lower(),
        )

        cached = self._multilingual_query_cache.get(
            cache_key
        )

        if cached is not None:

            return cached

        deterministic_query = (
            self._deterministic_multilingual_relation_query(
                question
            )
        )

        if deterministic_query:
            deterministic_query = self._clean_retrieval_query(
                deterministic_query
            )
            self._multilingual_query_cache[cache_key] = deterministic_query

            evidence_logger.record_event(
                event_name="MULTILINGUAL RELATION NORMALIZATION",
                status="DETERMINISTIC",
                details={
                    "original_question": question,
                    "canonical_query": deterministic_query,
                },
            )

            return deterministic_query

        prompt = (
            MULTILINGUAL_RETRIEVAL_QUERY_PROMPT.format(
                current_topic=(
                    current_topic
                    or "None"
                ),
                history=(
                    history
                    or "None"
                ),
                question=question
            )
        )

        try:

            auxiliary_client = self._auxiliary_llm_for_question(
                question
            )
            english_query = self._generate_with_latency(
                auxiliary_client,
                prompt,
                purpose="multilingual_retrieval_query",
            )

            english_query = self._clean_retrieval_query(
                english_query
            )

        except Exception as error:

            evidence_logger.record_error(
                location=(
                    "AnswerService."
                    "_build_english_retrieval_query"
                ),
                error=error,
                details={
                    "question":
                        question,
                }
            )

            english_query = ""

        self._multilingual_query_cache[
            cache_key
        ] = english_query

        return english_query

    def _combine_retrieval_queries(
        self,
        original_query: str,
        english_query: str
    ):

        """
        Combine original-language and English retrieval queries
        while avoiding exact duplicates.
        """

        queries = []

        for query in (
            original_query,
            english_query,
        ):

            clean_query = re.sub(
                r"\s+",
                " ",
                str(
                    query
                    or ""
                )
            ).strip()

            if not clean_query:

                continue

            if any(
                clean_query.lower()
                == existing.lower()
                for existing in queries
            ):

                continue

            queries.append(
                clean_query
            )

        return " | ".join(
            queries
        )

    @staticmethod
    def _semantic_relation_target(
        resolved_question: str,
        english_retrieval_query: str = "",
    ):

        """Prefer a canonical English relation target when translation exists."""

        english = re.sub(
            r"\s+",
            " ",
            str(english_retrieval_query or ""),
        ).strip()

        if english:
            return english

        return re.sub(
            r"\s+",
            " ",
            str(resolved_question or ""),
        ).strip()


    def _build_misra_compliance_prompt(
        self,
        context: str,
        question: str,
        history: str,
        results=None,
    ) -> str:
        generation_contract = MisraComplianceMode.build_generation_contract(
            question,
            results or [],
        )
        user_prompt = MISRA_COMPLIANCE_TEMPLATE.format(
            context=context,
            history=history,
            question=question,
        )
        if generation_contract:
            user_prompt = (
                generation_contract
                + "\n\n"
                + user_prompt
            )
        return SYSTEM_PROMPT + "\n\n" + user_prompt

    def _retrieve_context_for_request(
        self,
        search_question: str,
        semantic_target_question: str,
        misra_compliance_mode: bool,
    ):
        if not misra_compliance_mode:
            context, results = self.query_service.retrieve_context(
                search_question,
                intent_question=semantic_target_question,
            )
            return self._augment_reason_cross_reference_context(
                context=context,
                results=results,
                question=semantic_target_question,
            )

        # v6.4.62: explicit Rule/Directive explanation/rationale/example requests
        # must keep the authoritative parent Rule body as the citable anchor.  A
        # child rationale/example chunk may rank higher semantically, but it must
        # never replace the named Rule/Directive evidence.
        try:
            authoritative_records = MisraComplianceMode.load_authoritative_bm25_records()
            exact_evidence = (
                []
                if "PRIOR GROUNDED REFERENCE:" in str(semantic_target_question or "")
                else MisraComplianceMode.exact_reference_evidence(
                    authoritative_records,
                    semantic_target_question,
                )
            )
            if exact_evidence:
                evidence_logger.record_event(
                    event_name="MISRA EVIDENCE PLANNER",
                    status="EXACT REFERENCE ANCHOR",
                    details={
                        "matches": len(exact_evidence),
                        "references": [
                            str((item.get("metadata", {}) or {}).get("section_title", ""))
                            for item in exact_evidence
                        ],
                    },
                )
                exact_context = MisraComplianceMode.build_grounded_context(
                    exact_evidence,
                    semantic_target_question,
                )
                return self._augment_reason_cross_reference_context(
                    context=exact_context,
                    results=exact_evidence,
                    question=semantic_target_question,
                )
        except Exception as exact_planner_error:
            evidence_logger.record_event(
                event_name="MISRA EVIDENCE PLANNER",
                status="EXACT REFERENCE PLANNER SKIPPED",
                details={"error": str(exact_planner_error)},
            )

        # v6.4.57: explicit structured-family/list requests must reach the
        # retriever's complete structured-family path before the older semantic
        # MISRA rule-body rescue.  The rescue is intentionally bounded by a
        # small Top-K for assessment questions; using it first on a family list
        # can silently truncate an otherwise complete structured inventory.
        #
        # Call the retriever's source-grounded family detector directly rather
        # than running a second general hybrid search.  The detector itself
        # decides whether this is a supported structured-family request; when
        # it is not, the existing rescue/hybrid path continues unchanged.
        try:
            family_retriever = self.query_service._get_retriever()
            structured_family = family_retriever._retrieve_structured_rule_topic_family(
                query=search_question,
                intent_query=semantic_target_question,
            )
            if structured_family:
                structured_family = MisraComplianceMode.annotate_structured_family_assessment(
                    semantic_target_question,
                    structured_family,
                )
                evidence_logger.record_event(
                    event_name="MISRA STRUCTURED FAMILY PRE-RESCUE",
                    status="MATCHED COMPLETE STRUCTURED FAMILY",
                    details={
                        "matches": len(structured_family),
                        "references": [
                            str((item.get("metadata", {}) or {}).get("rule_id", ""))
                            for item in structured_family
                        ],
                    },
                )
                return (
                    MisraComplianceMode.build_grounded_context(structured_family, semantic_target_question),
                    structured_family,
                )
        except Exception as family_probe_error:
            evidence_logger.record_event(
                event_name="MISRA STRUCTURED FAMILY PRE-RESCUE",
                status="NO STRUCTURED FAMILY MATCH",
                details={"error": str(family_probe_error)},
            )

        # v6.5.0 primary natural-language path: resolve the question against
        # Rule/Directive semantic profiles derived from the indexed MISRA source.
        # This is corpus-driven (requirement + rationale + examples), not a
        # phrase-to-Rule table.  Ambiguous matches deliberately fall through to
        # MultiQuery/hybrid retrieval so recall can expand without sacrificing
        # precision.
        semantic_resolution = None
        semantic_resolution_attempted = False
        semantic_resolution_available = False
        semantic_scope = MisraComplianceMode.semantic_rule_resolution_scope(
            semantic_target_question,
            current_topic="MISRA",
        )
        if semantic_scope:
            semantic_resolution_attempted = True
            try:
                semantic_retriever = self.query_service._get_retriever()
                semantic_resolution = semantic_retriever.resolve_misra_rule_semantics(
                    semantic_target_question
                )
                semantic_resolution_available = str(
                    semantic_resolution.get("status", "")
                ).casefold() != "unavailable"
            except Exception as semantic_error:
                semantic_resolution = {
                    "status": "unavailable",
                    "accepted": False,
                    "error": str(semantic_error),
                }

            if semantic_resolution.get("accepted"):
                records = MisraComplianceMode.load_authoritative_bm25_records()
                semantic_evidence = MisraComplianceMode.authoritative_semantic_reference_evidence(
                    records,
                    kind=str(semantic_resolution.get("kind", "")),
                    identifier=str(semantic_resolution.get("identifier", "")),
                    diagnostics=semantic_resolution,
                )
                if semantic_evidence:
                    evidence_logger.record_event(
                        event_name="MISRA CORPUS SEMANTIC EVIDENCE",
                        status="AUTHORITATIVE RULE VERIFIED",
                        details={
                            "reference": semantic_resolution.get("reference", ""),
                            "semantic_score": semantic_resolution.get("semantic_score"),
                            "semantic_margin": semantic_resolution.get("semantic_margin"),
                            "similarity_margin": semantic_resolution.get("similarity_margin"),
                            "rerank_score": semantic_resolution.get("top_rerank_score"),
                            "rerank_margin": semantic_resolution.get("rerank_margin"),
                            "acceptance_basis": semantic_resolution.get("acceptance_basis", ""),
                            "bge_top_reference": semantic_resolution.get("bge_top_reference", ""),
                            "bge_top_score": semantic_resolution.get("bge_top_score"),
                            "full_multi_query_consensus": semantic_resolution.get("full_multi_query_consensus"),
                        },
                    )
                    return (
                        MisraComplianceMode.build_grounded_context(
                            semantic_evidence, semantic_target_question
                        ),
                        semantic_evidence,
                    )
                evidence_logger.record_event(
                    event_name="MISRA CORPUS SEMANTIC EVIDENCE",
                    status="AUTHORITATIVE REFERENCE VERIFICATION FAILED",
                    details={"reference": semantic_resolution.get("reference", "")},
                )

            if semantic_resolution_available:
                evidence_logger.record_event(
                    event_name="MISRA LEGACY PHRASE RESCUE",
                    status="SKIPPED FOR NATURAL QUERY",
                    details={
                        "semantic_status": semantic_resolution.get("status", ""),
                        "reason": (
                            "The corpus semantic index was available. Ambiguous/no-match "
                            "natural questions continue to MultiQuery instead of being "
                            "forced through a phrase-specific Rule mapping."
                        ),
                    },
                )

        # First try a deterministic MISRA-only structured rule-body rescue.
        # v6.4.28 proved that the correct Rule bodies were present in the ZIP,
        # but the Windows CLI run still returned zero chunks.  Read the same
        # authoritative on-disk BM25 corpus directly before touching any
        # process-local/Streamlit-cached retriever object.  This keeps the
        # rescue deterministic across CLI, Streamlit and LAN runtime modes.
        rescued = []
        rescue_diagnostics = {}
        allow_legacy_phrase_rescue = (
            not semantic_resolution_attempted
            or not semantic_resolution_available
            or MisraComplianceMode.looks_like_c_cpp(semantic_target_question)
        )
        if allow_legacy_phrase_rescue:
            try:
                rescued, rescue_diagnostics = (
                    MisraComplianceMode.rule_body_cue_rescue_from_authoritative_corpus(
                        question=semantic_target_question,
                        top_k=6,
                    )
                )
            except Exception as rescue_error:
                rescue_diagnostics = {
                    "authoritative_corpus_error": str(rescue_error),
                    "semantic_cues": MisraComplianceMode.semantic_cues(
                        semantic_target_question
                    ),
                }
        else:
            rescue_diagnostics = {
                "semantic_status": (semantic_resolution or {}).get("status", ""),
                "semantic_cues": [],
                "legacy_phrase_rescue": "skipped",
            }

        if rescued:
            evidence_logger.record_event(
                event_name="MISRA RULE-BODY CUE RESCUE",
                status="MATCHED AUTHORITATIVE CORPUS",
                details={
                    **rescue_diagnostics,
                    "matches": len(rescued),
                    "cues": [item.get("_misra_cue", "") for item in rescued],
                },
            )
            return MisraComplianceMode.build_grounded_context(rescued, semantic_target_question), rescued

        # Compatibility fallback: if the authoritative corpus cannot be read,
        # try the already-initialized BM25 resource before using general hybrid
        # retrieval.  This does not change the global retrieval threshold.
        if allow_legacy_phrase_rescue:
            try:
                retriever = self.query_service._get_retriever()
                cached_records = getattr(getattr(retriever, "bm25", None), "records", [])
                rescued = MisraComplianceMode.rule_body_cue_rescue(
                    records=cached_records,
                    question=semantic_target_question,
                    top_k=6,
                )
                rescue_diagnostics["cached_record_count"] = len(cached_records or [])
                rescue_diagnostics["cached_matches"] = [
                    str((item.get("metadata", {}) or {}).get("section_title", ""))
                    for item in rescued
                ]
            except Exception as rescue_error:
                rescue_diagnostics["cached_rescue_error"] = str(rescue_error)
                rescued = []
        else:
            rescued = []

        if rescued:
            evidence_logger.record_event(
                event_name="MISRA RULE-BODY CUE RESCUE",
                status="MATCHED CACHED CORPUS",
                details={
                    **rescue_diagnostics,
                    "matches": len(rescued),
                    "cues": [item.get("_misra_cue", "") for item in rescued],
                },
            )
            return MisraComplianceMode.build_grounded_context(rescued, semantic_target_question), rescued

        evidence_logger.record_event(
            event_name="MISRA RULE-BODY CUE RESCUE",
            status="NO MATCH",
            details={
                **rescue_diagnostics,
                "note": (
                    "Falling back to the existing source-scoped hybrid + reranker path; "
                    "the global retrieval threshold remains unchanged."
                ),
            },
        )

        try:
            rerank_intent_question = MisraComplianceMode.build_rerank_query(
                semantic_target_question
            )
            if rerank_intent_question != str(semantic_target_question or "").strip():
                evidence_logger.record_event(
                    event_name="MISRA RERANK INTENT",
                    status="SOURCE-LANGUAGE CANONICALIZED",
                    details={
                        "original_intent": semantic_target_question,
                        "rerank_intent": rerank_intent_question,
                    },
                )
            context, results = self.query_service.retrieve_context(
                search_question,
                intent_question=rerank_intent_question,
                source_family="misra",
                final_top_k_override=6,
            )
            return MisraComplianceMode.build_grounded_context(results, semantic_target_question), results
        except TypeError as error:
            # Compatibility with lightweight QA doubles that still expose the
            # historical two-argument retrieve_context signature. Production
            # QueryService supports the scoped arguments above.
            if "unexpected keyword" not in str(error):
                raise
            context, results = self.query_service.retrieve_context(
                search_question,
                intent_question=semantic_target_question,
            )
            results = MisraComplianceMode.filter_misra_results(results)[:6]
            return MisraComplianceMode.build_grounded_context(results, semantic_target_question), results

    @staticmethod
    def _post_retrieval_misra_authoritative_promotion(
        question: str,
        results,
    ):
        """Promote accepted structured MISRA evidence to deterministic use.

        This is a downstream safety net, not a threshold bypass.  It runs only
        for direct requirement/Yes-No concepts that already have accepted
        structured Rule/Directive evidence from normal retrieval.  The same
        semantic cue must independently resolve against the authoritative BM25
        corpus, and at least one authoritative reference must overlap the
        references already present in the accepted result set.

        In other words: retrieval must have found the Rule, and the source-backed
        concept resolver must agree on that same Rule.  Only then do we replace
        the generic mixed context with the authoritative Rule-body evidence so a
        later LLM cannot turn a proven answer into "Information not found".
        """

        raw = re.sub(r"\s+", " ", str(question or "")).strip()
        if not raw or not results:
            return "", [], {}

        direct_intent = bool(
            MisraComplianceMode.yes_no_intent(raw)
            or MisraComplianceMode.natural_requirement_lookup_intent(raw)
        )
        if not direct_intent:
            return "", [], {}

        semantic_cues = MisraComplianceMode.semantic_cues(raw)
        if not semantic_cues:
            return "", [], {}

        accepted_refs = MisraComplianceMode.available_references(results)
        if not accepted_refs:
            return "", [], {}

        try:
            rescued, diagnostics = (
                MisraComplianceMode.rule_body_cue_rescue_from_authoritative_corpus(
                    question=raw,
                    top_k=6,
                )
            )
        except Exception as error:
            return "", [], {
                "error": str(error),
                "semantic_cues": semantic_cues,
                "accepted_references": sorted(accepted_refs),
            }

        rescued_refs = MisraComplianceMode.available_references(rescued)
        overlap = accepted_refs.intersection(rescued_refs)
        if not overlap:
            return "", [], {
                **(diagnostics or {}),
                "semantic_cues": semantic_cues,
                "accepted_references": sorted(accepted_refs),
                "authoritative_references": sorted(rescued_refs),
                "overlap": [],
            }

        promoted = []
        for item in rescued:
            item_refs = MisraComplianceMode.available_references([item])
            if item_refs.intersection(overlap):
                promoted.append(item)

        if not promoted:
            return "", [], {}

        context = MisraComplianceMode.build_grounded_context(promoted, raw)
        return context, promoted, {
            **(diagnostics or {}),
            "semantic_cues": semantic_cues,
            "accepted_references": sorted(accepted_refs),
            "authoritative_references": sorted(rescued_refs),
            "overlap": sorted(overlap),
            "promoted_count": len(promoted),
        }


    @staticmethod
    def _explicit_external_knowledge_request(question: str) -> bool:
        """Return True when the user explicitly asks DocuBot to leave the KB.

        DocuBot is intentionally company-knowledge-only.  This guard fires only
        on explicit outside/general-knowledge instructions, so ordinary current
        dates or words such as "latest" do not bypass retrieval on their own.
        Short-circuiting here improves both hallucination safety and latency.
        """

        clean = re.sub(r"\s+", " ", str(question or "").strip().casefold())
        if not clean:
            return False
        outside_instruction = bool(re.search(
            r"\b(?:use|using|gamitin|gamit)\b.{0,40}\b(?:general|outside|external)\s+knowledge\b|"
            r"\b(?:externally|outside\s+(?:the\s+)?(?:company\s+)?(?:knowledge|documents|kb)|"
            r"from\s+the\s+(?:internet|web))\b",
            clean,
            re.IGNORECASE,
        ))
        explicit_missing_scope = bool(re.search(
            r"\b(?:not\s+in|outside|wala\s+sa|hindi\s+nasa)\b.{0,55}"
            r"\b(?:company\s+documents?|company\s+knowledge|knowledge\s+base|kb|documents?)\b",
            clean,
            re.IGNORECASE,
        ))
        override_language = bool(re.search(
            r"\b(?:kahit|even\s+if|if\s+(?:it(?:'s| is)?|that(?:'s| is)?)\s+not)\b.{0,55}"
            r"\b(?:externally|general\s+knowledge|outside\s+knowledge)\b",
            clean,
            re.IGNORECASE,
        ))
        return outside_instruction or (explicit_missing_scope and override_language)

    @staticmethod
    def _technical_profile_fail_closed_request(question: str) -> bool:
        """Fast-fail clearly unsupported live/general-knowledge requests.

        The Option-C technical corpus is static and MISRA-focused.  Current-world
        questions should not spend tens of seconds embedding/reranking unrelated
        technical chunks only to return the same fallback.  Keep this guard narrow
        and profile-specific so legacy/general corpora retain their old behavior.
        """
        if str(KNOWLEDGE_PROFILE or "").casefold() != "technical":
            return False
        clean = re.sub(r"\s+", " ", str(question or "").strip().casefold())
        if not clean:
            return False
        freshness = bool(re.search(
            r"\b(?:latest|newest|current|today|this\s+year|most\s+recent)\b",
            clean,
        ))
        misra_release = bool(
            "misra" in clean
            and re.search(r"\b(?:rule|directive|release|released|published|added|version|edition)\b", clean)
        )
        public_current_role = bool(re.search(
            r"\bwho\s+is\s+(?:the\s+)?current\s+(?:president|prime\s+minister|king|queen)\b",
            clean,
        ))
        return (freshness and misra_release) or public_current_role

    @staticmethod
    def _source_results_supporting_answer(results, answer: str):
        """Select accepted result(s) that actually contain deterministic facts.

        A reranker can place a semantically noisy document above the document
        whose explicit labeled facts were used by a deterministic relation
        finalizer.  Source display must follow the facts, not ranking alone.
        """

        if not results or not answer:
            return []

        fact_lines = []
        for raw_line in str(answer).splitlines():
            line = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", raw_line).strip()
            line = re.sub(r"^\*{0,2}(?:Approval|Eligibility|Scope|Vacation Leave|Sick Leave)\s*:\*{0,2}\s*", "", line, flags=re.I)
            line = re.sub(r"\s+", " ", line).strip()
            if len(line) >= 18 and not line.casefold().startswith("based on these two statements"):
                fact_lines.append(line)

        if not fact_lines:
            return []

        scored = []
        for index, item in enumerate(results):
            if not isinstance(item, dict):
                continue
            text = re.sub(r"\s+", " ", str(item.get("text", "") or "")).casefold()
            score = sum(1 for fact in fact_lines if fact.casefold() in text)
            if score:
                scored.append((score, -index, item))

        if not scored:
            return []
        best = max(score for score, _neg, _item in scored)
        return [item for score, _neg, item in scored if score == best]

    def ask(
        self,
        question
    ):

        request_started = time.perf_counter()
        retrieval_elapsed = 0.0

        history = (
            self._build_chat_history()
        )

        # ======================================
        # Normalize Question
        # ======================================
        normalized_question = (
            self.query_normalizer.normalize(
                question
            )
        )

        # Previous messages only
        messages = (
            ChatManager.get_current_messages()
        )

        previous_messages = messages[:-1]

        # ======================================
        # Reset topic when this is a new chat
        # ======================================
        if not previous_messages:

            ChatManager.set_current_topic(
                None
            )

        # Preserve the last valid conversation topic.
        previous_topic = (
            ChatManager.get_current_topic()
        )

        if KNOWLEDGE_PROFILE == "technical" and has_malformed_structured_reference(question):
            evidence_logger.record_event(
                event_name="EXACT STRUCTURED REFERENCE GUARD",
                status="MALFORMED IDENTIFIER; STOPPED",
                details={"question": str(question or "")},
            )
            evidence_logger.record_event(
                event_name="REQUEST LATENCY PROFILE",
                status="FALLBACK BEFORE RETRIEVAL",
                details={
                    "retrieval_seconds": 0.0,
                    "total_seconds": round(time.perf_counter() - request_started, 4),
                },
            )
            evidence_logger.record_answer(
                final_answer=NO_RESULT_MESSAGE,
                fallback_used=True,
                sources=[],
            )
            return {"answer": NO_RESULT_MESSAGE, "sources": [], "chunks": []}

        if self._explicit_external_knowledge_request(question) or self._technical_profile_fail_closed_request(question):
            evidence_logger.record_event(
                event_name="COMPANY-KB BOUNDARY",
                status=(
                    "EXPLICIT EXTERNAL KNOWLEDGE REQUEST REJECTED"
                    if self._explicit_external_knowledge_request(question)
                    else "TECHNICAL PROFILE FAST FAIL-CLOSED"
                ),
                details={"question": str(question or "")},
            )
            evidence_logger.record_event(
                event_name="REQUEST LATENCY PROFILE",
                status="FALLBACK BEFORE RETRIEVAL",
                details={
                    "retrieval_seconds": 0.0,
                    "total_seconds": round(time.perf_counter() - request_started, 4),
                },
            )
            evidence_logger.record_answer(
                final_answer=NO_RESULT_MESSAGE,
                fallback_used=True,
                sources=[],
            )
            return {"answer": NO_RESULT_MESSAGE, "sources": [], "chunks": []}

        compiler_clarification = self._compiler_switch_ambiguity_clarification(
            question
        )
        if compiler_clarification:
            evidence_logger.record_question_pipeline(
                original_question=question,
                normalized_question=normalized_question,
                resolved_question=normalized_question,
                search_question=normalized_question,
                current_topic=(previous_topic or ""),
                history=history,
            )
            evidence_logger.record_event(
                event_name="AMBIGUOUS COMPILER TERMINOLOGY",
                status="CLARIFICATION REQUIRED",
                details={
                    "question": str(question or ""),
                    "reason": (
                        "Generic compiler switch/flag wording is not treated as "
                        "synonymous with compiler configuration."
                    ),
                },
            )
            evidence_logger.record_event(
                event_name="REQUEST LATENCY PROFILE",
                status="CLARIFICATION BEFORE RETRIEVAL",
                details={
                    "retrieval_seconds": 0.0,
                    "total_seconds": round(time.perf_counter() - request_started, 4),
                },
            )
            evidence_logger.record_answer(
                final_answer=compiler_clarification,
                fallback_used=False,
                sources=[],
            )
            return {
                "answer": compiler_clarification,
                "sources": [],
                "chunks": [],
            }

        previous_grounded_state = ChatManager.get_grounded_state()
        grounded_followup = self._is_grounded_followup_candidate(
            question,
            previous_grounded_state,
        )

        misra_compliance_mode = MisraComplianceMode.is_request(
            question,
            current_topic=previous_topic or "",
        )

        # v6.5.0: text-only MISRA standards questions may contain completely
        # unseen wording and therefore must not depend on the legacy phrase cue
        # catalog just to enter the source-grounded path.  This gate identifies
        # MISRA scope only; Rule selection is performed later by the corpus-
        # derived semantic resolver.
        semantic_rule_scope = MisraComplianceMode.semantic_rule_resolution_scope(
            question,
            current_topic=previous_topic or "",
        )
        if semantic_rule_scope and not misra_compliance_mode:
            misra_compliance_mode = True
            evidence_logger.record_event(
                event_name="MISRA CORPUS SEMANTIC SCOPE",
                status="DETECTED",
                details={
                    "reason": (
                        "Text-only MISRA question entered Rule-level semantic resolution "
                        "without requiring a phrase-to-Rule mapping."
                    )
                },
            )

        # A short deictic follow-up (for example "Bakit mo nasabi yan?")
        # may not contain MISRA/code keywords of its own.  If it explicitly
        # refers to the immediately prior accepted MISRA assessment, preserve
        # that mode so the prior user code/scenario can be reused safely.
        if (
            grounded_followup
            and isinstance(previous_grounded_state, dict)
            and previous_grounded_state.get("misra") is True
        ):
            misra_compliance_mode = True

        if misra_compliance_mode:
            evidence_logger.record_event(
                event_name="MISRA NATURAL COMPLIANCE MODE",
                status="DETECTED",
                details={
                    "intent_based": True,
                    "code_like": MisraComplianceMode.looks_like_c_cpp(question),
                },
            )

        # ======================================
        # Resolve Follow-up References
        # ======================================
        resolved_question = (
            self.conversation_resolver.resolve(
                previous_messages,
                normalized_question
            )
        )

        grounded_anchor_question = ""
        if grounded_followup and isinstance(previous_grounded_state, dict):
            grounded_anchor_question = str(
                previous_grounded_state.get("anchor_question", "") or ""
            ).strip()

        # ======================================
        # Enrich Question for Retrieval Only
        # ======================================
        original_search_question = (
            self.query_enricher.enrich(
                resolved_question,
                intent_question=question
            )
        )

        if grounded_anchor_question and not misra_compliance_mode:
            anchor_normalized = self.query_normalizer.normalize(
                grounded_anchor_question
            )
            anchor_search = self.query_enricher.enrich(
                anchor_normalized,
                intent_question=grounded_anchor_question,
            )
            original_search_question = self._combine_retrieval_queries(
                original_search_question,
                anchor_search,
            )
            evidence_logger.record_event(
                event_name="GROUNDED FOLLOW-UP ANCHOR",
                status="RETRIEVAL AUGMENTED",
                details={
                    "anchor_question": grounded_anchor_question,
                    "current_question": question,
                },
            )

        english_retrieval_query = ""

        if (
            ENABLE_MULTILINGUAL_RETRIEVAL
            and not misra_compliance_mode
            and extract_structured_reference(
                resolved_question or question
            ) is None
            and not self._is_clearly_english_query(
                question
            )
        ):

            english_retrieval_query = (
                self._build_english_retrieval_query(
                    question=question,
                    history=history,
                    current_topic=(
                        ChatManager.get_current_topic()
                        or ""
                    )
                )
            )

        english_search_question = ""

        if english_retrieval_query:

            normalized_english_query = (
                self.query_normalizer.normalize(
                    english_retrieval_query
                )
            )

            english_search_question = (
                self.query_enricher.enrich(
                    normalized_english_query,
                    intent_question=english_retrieval_query
                )
            )

            evidence_logger.record_event(
                event_name=(
                    "MULTILINGUAL RETRIEVAL QUERY"
                ),
                details={
                    "original_question":
                        question,

                    "english_query":
                        english_retrieval_query,

                    "enriched_english_query":
                        english_search_question,
                },
                status="CREATED"
            )

        search_question = (
            self._combine_retrieval_queries(
                original_search_question,
                english_search_question
            )
        )

        # For non-English questions, keep the user's original language for
        # response generation but use the canonical English retrieval query as
        # the semantic relation target. This avoids asking the model to match
        # English source facts against a noisy translated/normalized relation
        # string while preserving the original user-facing language.
        semantic_target_question = self._semantic_relation_target(
            resolved_question=resolved_question,
            english_retrieval_query=english_retrieval_query,
        )

        if grounded_anchor_question and not misra_compliance_mode:
            semantic_target_question = (
                f"{semantic_target_question} | Prior grounded request: "
                f"{grounded_anchor_question}"
            ).strip()

        if english_retrieval_query:
            evidence_logger.record_event(
                event_name="MULTILINGUAL SEMANTIC TARGET",
                status="APPLIED",
                details={
                    "resolved_question": resolved_question,
                    "semantic_target": semantic_target_question,
                },
            )

        # v6.4.99: a mixed-language technical question can be too loose for
        # the first-pass MISRA intent classifier, yet its normalized English
        # retrieval query can expose an exact source-language requirement cue.
        # Promote only explicit/inherited MISRA turns with such a cue back into
        # the deterministic MISRA evidence path *before* MultiQuery/reranking.
        # This preserves general multilingual behavior for non-MISRA questions.
        post_normalized_misra_query = ""
        if not misra_compliance_mode and english_retrieval_query:
            post_normalized_misra_query = (
                MisraComplianceMode.post_normalization_rescue_query(
                    original_question=question,
                    normalized_query=english_retrieval_query,
                    current_topic=(ChatManager.get_current_topic() or ""),
                )
            )
        if post_normalized_misra_query:
            misra_compliance_mode = True
            evidence_logger.record_event(
                event_name="MISRA POST-NORMALIZATION RESCUE",
                status="DETERMINISTIC MODE PROMOTED",
                details={
                    "original_question": question,
                    "normalized_query": post_normalized_misra_query,
                    "semantic_cues": MisraComplianceMode.semantic_cues(
                        post_normalized_misra_query
                    ),
                    "reason": (
                        "The normalized English query exposed a source-backed "
                        "MISRA requirement cue; authoritative Rule-body rescue "
                        "will run before MultiQuery/reranker fallback."
                    ),
                },
            )

        misra_evidence_question = (
            post_normalized_misra_query
            or str(question or "").strip()
        )
        if (
            misra_compliance_mode
            and grounded_anchor_question
            and not self._question_has_visible_code(question)
        ):
            prior_references = []
            if isinstance(previous_grounded_state, dict):
                prior_references = list(previous_grounded_state.get("references") or [])
            reference_hint = ""
            if (
                len(prior_references) == 1
                and extract_structured_reference(question) is None
            ):
                reference_hint = f"\nPRIOR GROUNDED REFERENCE: {prior_references[0]}"

            misra_evidence_question = (
                f"{question.strip()}\n\n"
                f"PRIOR USER CODE/SCENARIO FOR THIS FOLLOW-UP:\n"
                f"{grounded_anchor_question}"
                f"{reference_hint}"
            ).strip()
            evidence_logger.record_event(
                event_name="GROUNDED FOLLOW-UP ANCHOR",
                status="MISRA CODE/SCENARIO REUSED",
                details={
                    "anchor_question": grounded_anchor_question,
                    "current_question": question,
                },
            )

        if misra_compliance_mode:
            search_question = MisraComplianceMode.build_search_query(
                question=misra_evidence_question,
                resolved_question=resolved_question,
            )
            # Use the anchored code/scenario only for deterministic cue
            # interpretation/retrieval. The final prompt still receives the
            # user's current question plus normal conversation history.
            semantic_target_question = misra_evidence_question

        evidence_logger.record_question_pipeline(
            original_question=question,
            normalized_question=normalized_question,
            resolved_question=resolved_question,
            search_question=search_question,
            current_topic=(
                ChatManager.get_current_topic()
                or ""
            ),
            history=history
        )

        # ======================================
        # DEBUG
        # ======================================
        if DEBUG_MODE:

            print("=" * 60)
            print("Conversation Debug")
            print("=" * 60)

            print(
                f"Original Question   : "
                f"{question}"
            )

            print(
                f"Normalized Question : "
                f"{normalized_question}"
            )

            print(
                f"Resolved Question   : "
                f"{resolved_question}"
            )

            print(
                f"Search Question     : "
                f"{search_question}"
            )

            print("\nCurrent Topic:")

            print(
                ChatManager.get_current_topic()
            )

            print("\nHistory:")

            print(history)

            print("=" * 60)

        # ======================================
        # Retrieve Context
        # ======================================
        retrieval_retry_used = False
        retrieval_started = time.perf_counter()
        context, results = self._retrieve_context_for_request(
            search_question=search_question,
            semantic_target_question=semantic_target_question,
            misra_compliance_mode=misra_compliance_mode,
        )
        retrieval_elapsed = time.perf_counter() - retrieval_started

        # For an explicit same-chat relational follow-up, reject a fresh result
        # set that completely loses the previously accepted source.  This avoids
        # short phrases such as "sino naman ang nag-aapprove?" drifting to an
        # unrelated high-frequency document.  No extra retrieval/model call is
        # added; we reuse only already accepted evidence from this chat.
        if (
            grounded_followup
            and not misra_compliance_mode
            and isinstance(previous_grounded_state, dict)
            and previous_grounded_state.get("context")
            and previous_grounded_state.get("results")
        ):
            def _source_keys(items):
                keys = set()
                for value in items or []:
                    if not isinstance(value, dict):
                        continue
                    meta = value.get("metadata", {}) or {}
                    key = str(meta.get("file_path") or meta.get("file_name") or "").strip().casefold()
                    if key:
                        keys.add(key)
                return keys

            prior_keys = _source_keys(previous_grounded_state.get("results"))
            fresh_keys = _source_keys(results)
            if prior_keys and (not fresh_keys or prior_keys.isdisjoint(fresh_keys)):
                context = str(previous_grounded_state.get("context") or "")
                results = list(previous_grounded_state.get("results") or [])
                evidence_logger.record_event(
                    event_name="GROUNDED FOLLOW-UP CONTEXT",
                    status="REUSED AFTER SOURCE DRIFT",
                    details={
                        "current_question": question,
                        "anchor_question": grounded_anchor_question,
                        "prior_sources": sorted(prior_keys),
                        "fresh_sources": sorted(fresh_keys),
                    },
                )

        # v6.4.101: evidence-aware MISRA promotion.  A narrow source-backed
        # concept can occasionally miss the first intent gate even though the
        # user explicitly asked about MISRA.  Before paying for another full
        # multilingual MultiQuery pass, probe the authoritative Rule/Directive
        # corpus using semantic cues visible in the original and normalized
        # query.  This restores useful evidence that the generic reranker may
        # have discarded without weakening the global 0.55 threshold.
        misra_scope_hint = (
            "misra" in str(question or "").casefold()
            or "misra" in str(previous_topic or "").casefold()
        )
        evidence_promotion_probe = self._combine_retrieval_queries(
            str(question or "").strip(),
            str(english_retrieval_query or "").strip(),
        )
        if not context and not misra_compliance_mode and misra_scope_hint:
            promotion_cues = MisraComplianceMode.semantic_cues(
                evidence_promotion_probe
            )
            if promotion_cues:
                try:
                    promoted_results, promotion_diagnostics = (
                        MisraComplianceMode.rule_body_cue_rescue_from_authoritative_corpus(
                            question=evidence_promotion_probe,
                            top_k=6,
                        )
                    )
                except Exception as promotion_error:
                    promoted_results = []
                    promotion_diagnostics = {"error": str(promotion_error)}

                if promoted_results:
                    context = MisraComplianceMode.build_grounded_context(
                        promoted_results,
                        evidence_promotion_probe,
                    )
                    results = promoted_results
                    misra_compliance_mode = True
                    semantic_target_question = evidence_promotion_probe
                    evidence_logger.record_event(
                        event_name="MISRA EVIDENCE-AWARE PROMOTION",
                        status="AUTHORITATIVE EVIDENCE PROMOTED",
                        details={
                            **promotion_diagnostics,
                            "semantic_cues": promotion_cues,
                            "matches": len(promoted_results),
                            "reason": (
                                "Generic retrieval returned no accepted context, but "
                                "the explicit MISRA turn had source-backed semantic cues. "
                                "Authoritative evidence was promoted before any duplicate "
                                "multilingual MultiQuery retry."
                            ),
                        },
                    )
                else:
                    evidence_logger.record_event(
                        event_name="MISRA EVIDENCE-AWARE PROMOTION",
                        status="NO AUTHORITATIVE MATCH",
                        details={
                            **promotion_diagnostics,
                            "semantic_cues": promotion_cues,
                        },
                    )

        # A bilingual query may occasionally dilute reranker relevance.
        # When it returns no accepted context, retry using only the
        # canonical English query. This keeps English performance stable
        # while allowing non-English questions to retrieve English documents.
        # v6.4.101 suppresses a duplicate retry for explicit MISRA turns when
        # that same English query was already included in the first combined
        # retrieval.  This removes the pathological second 12-15 s MultiQuery
        # generation seen in failed Taglish tests without changing ordinary
        # multilingual retrieval behavior.
        duplicate_misra_retry = bool(
            misra_scope_hint
            and english_search_question
            and english_search_question.casefold() in str(search_question or "").casefold()
        )

        if (
            not context
            and not misra_compliance_mode
            and MULTILINGUAL_RETRY_ON_EMPTY
            and english_search_question
            and english_search_question.lower()
            != search_question.lower()
            and not duplicate_misra_retry
        ):

            evidence_logger.record_event(
                event_name=(
                    "MULTILINGUAL RETRIEVAL RETRY"
                ),
                details={
                    "first_query":
                        search_question,

                    "retry_query":
                        english_search_question,
                },
                status="STARTED"
            )

            retry_started = time.perf_counter()
            context, results = (
                self.query_service
                .retrieve_context(
                    english_search_question,
                    intent_question=semantic_target_question
                )
            )
            retrieval_elapsed += time.perf_counter() - retry_started

            if context:

                retrieval_retry_used = True

                evidence_logger.record_event(
                    event_name=(
                        "MULTILINGUAL RETRIEVAL RETRY"
                    ),
                    details=(
                        "English-only retrieval returned context."
                    ),
                    status="SUCCESS"
                )

            else:

                evidence_logger.record_event(
                    event_name=(
                        "MULTILINGUAL RETRIEVAL RETRY"
                    ),
                    details=(
                        "English-only retrieval also returned no context."
                    ),
                    status="NO CONTEXT"
                )

        if (
            not context
            and not misra_compliance_mode
            and MULTILINGUAL_RETRY_ON_EMPTY
            and duplicate_misra_retry
        ):
            evidence_logger.record_event(
                event_name="MULTILINGUAL RETRIEVAL RETRY",
                status="SKIPPED DUPLICATE MISRA QUERY",
                details={
                    "first_query": search_question,
                    "retry_query": english_search_question,
                    "reason": (
                        "The English retrieval query was already included in the "
                        "first combined MISRA retrieval; repeating MultiQuery would "
                        "add latency without adding a new semantic target."
                    ),
                },
            )

        # v6.4.102: downstream authoritative finalization safety net.
        # If normal retrieval already accepted a structured MISRA Rule/Directive
        # that agrees with the source-backed semantic concept, promote that exact
        # Rule body before answer routing.  This is specifically designed to
        # prevent a later generative step from erasing evidence that retrieval
        # already proved.  No global score threshold is changed.
        if context and not misra_compliance_mode:
            (
                promoted_context,
                promoted_results,
                post_retrieval_promotion_details,
            ) = self._post_retrieval_misra_authoritative_promotion(
                question=question,
                results=results,
            )
            if promoted_context and promoted_results:
                context = promoted_context
                results = promoted_results
                misra_compliance_mode = True
                misra_evidence_question = str(question or "").strip()
                semantic_target_question = misra_evidence_question
                evidence_logger.record_event(
                    event_name="MISRA POST-RETRIEVAL AUTHORITATIVE FINALIZATION",
                    status="PROMOTED BEFORE GENERATION",
                    details={
                        **post_retrieval_promotion_details,
                        "reason": (
                            "Normal retrieval already accepted structured MISRA "
                            "evidence and the authoritative concept resolver agreed "
                            "on the same Rule/Directive; deterministic finalization "
                            "therefore takes precedence over generic LLM generation."
                        ),
                    },
                )

        if (
            misra_compliance_mode
            and context
            and not MisraComplianceMode.supports_requested_standard(
                question, results
            )
        ):
            evidence_logger.record_event(
                event_name="MISRA STANDARD FAMILY GUARD",
                status="UNSUPPORTED REQUESTED FAMILY",
                details={
                    "requested_family": MisraComplianceMode.requested_standard_family(question),
                },
            )
            context, results = "", []

        # ======================================
        # DEBUG
        # ======================================
        if DEBUG_MODE:

            print("\nRetrieved Context:\n")
            print(context)

            print("=" * 60)

            print(
                f"Retrieved Chunks : "
                f"{len(results)}"
            )

            print("Retrieved Sources:")

            for item in results:

                print(
                    "-",
                    item["metadata"].get(
                        "file_name",
                        "Unknown"
                    )
                )

            print("=" * 60)

        # Conservative same-chat follow-up rescue: if the new turn explicitly
        # refers to the immediately prior grounded answer/source and fresh
        # retrieval returns nothing, reuse only that accepted prior evidence.
        # This is not assistant-answer memory and does not introduce new facts.
        if (
            not context
            and grounded_followup
            and isinstance(previous_grounded_state, dict)
            and bool(previous_grounded_state.get("misra")) == bool(misra_compliance_mode)
            and previous_grounded_state.get("context")
            and previous_grounded_state.get("results")
        ):
            context = str(previous_grounded_state.get("context") or "")
            results = list(previous_grounded_state.get("results") or [])
            evidence_logger.record_event(
                event_name="GROUNDED FOLLOW-UP CONTEXT",
                status="REUSED AFTER EMPTY RETRIEVAL",
                details={
                    "current_question": question,
                    "anchor_question": grounded_anchor_question,
                    "result_count": len(results),
                },
            )

        # Return fallback message if nothing found.
        # Do not update topic memory.
        if not context:

            ChatManager.set_current_topic(
                previous_topic
            )

            if DEBUG_MODE:

                print(
                    "\n[TOPIC MEMORY NOT UPDATED] "
                    "No context found."
                )

            evidence_logger.record_event(
                event_name="REQUEST LATENCY PROFILE",
                status="FALLBACK BEFORE GENERATION",
                details={
                    "retrieval_seconds": round(retrieval_elapsed, 4),
                    "total_seconds": round(
                        time.perf_counter() - request_started,
                        4
                    ),
                },
            )

            evidence_logger.record_answer(
                final_answer=NO_RESULT_MESSAGE,
                fallback_used=True,
                sources=[]
            )

            return {

                "answer":
                    NO_RESULT_MESSAGE,

                "sources": [],

                "chunks": []
            }

        # ======================================
        # Route / Generate Final Answer
        # ======================================
        # Retrieval always happens first and remains the source-of-truth
        # boundary. Routing only decides how to turn accepted context into a
        # user-facing answer.
        final_question = question.strip()

        answer_focus = (
            MisraComplianceMode.answer_focus(final_question)
            if misra_compliance_mode
            else self._detect_answer_focus(
                final_question,
                semantic_target_question
            )
        )

        contextual_followup_prompt = bool(
            re.match(
                r"(?i)^\s*(?:does|do|is|are|why|how|which\s+part|what\s+other|"
                r"bakit|paano|aling\s+part|may\s+iba|kung\s+aayusin)\b",
                final_question,
            )
        )
        if (
            grounded_followup
            and not misra_compliance_mode
            and (
                answer_focus.startswith(("YES OR NO:", "GENERAL:"))
                or contextual_followup_prompt
            )
        ):
            answer_focus = (
                "GROUNDED EXPLANATION: Resolve the user's reference to the prior "
                "grounded request using only the currently accepted company context. "
                "Answer the follow-up directly, distinguish what the document "
                "explicitly states from what it does not establish, and do not turn "
                "a narrower scope statement into a broader claim."
                f' The requested relation or target is: "{semantic_target_question}".'
            )

        evidence_logger.record_event(
            event_name="ANSWER FOCUS",
            details=answer_focus,
            status="DETECTED"
        )

        precomputed_compound_answer = ""
        precomputed_comparison_answer = ""
        precomputed_temporal_answer = ""
        precomputed_relation_answer = ""
        precomputed_misra_guidance_answer = (
            MisraComplianceMode.deterministic_guidance(
                misra_evidence_question,
                results,
            )
            if (
                misra_compliance_mode
                and answer_focus.startswith("MISRA GUIDANCE:")
                and not LLM_CERTIFICATION_FORCE_GENERATION
            )
            else ""
        )
        precomputed_misra_requirement_lookup_answer = (
            MisraComplianceMode.deterministic_requirement_lookup(
                misra_evidence_question,
                results,
            )
            if (
                misra_compliance_mode
                and answer_focus.startswith("MISRA REQUIREMENT LOOKUP:")
                and not LLM_CERTIFICATION_FORCE_GENERATION
            )
            else ""
        )

        precomputed_structured_major_family_answer = (
            self._deterministic_structured_major_family_answer(results)
        )

        precomputed_structured_list_answer = (
            self._deterministic_structured_topic_list_answer(results)
            if answer_focus.startswith("LIST:")
            or (
                answer_focus.startswith("MISRA GUIDANCE:")
                and not precomputed_misra_guidance_answer
            )
            else ""
        )
        precomputed_named_section_answer = (
            self._deterministic_named_section_answer(results)
            if any(
                isinstance(item, dict) and item.get("_structured_section_title_anchor")
                for item in (results or [])
            )
            else ""
        )
        precomputed_structured_comparison_answer = (
            self._deterministic_structured_comparison_answer(results)
            if answer_focus.startswith("STRUCTURED COMPARISON:")
            else ""
        )
        precomputed_structured_multi_answer = (
            self._deterministic_structured_multi_reference_answer(
                results,
                question=semantic_target_question or final_question,
            )
            if answer_focus.startswith("STRUCTURED MULTI EXPLANATION:")
            else ""
        )

        # Exact Rule/Directive information requests are finalized directly from
        # the already accepted structured block. This prevents generic relation
        # extractors or grounding guards from mangling source-grounded summaries,
        # examples, and classifications.
        precomputed_structured_exact_answer = ""
        structured_exact_refs = [
            ref for ref in extract_structured_references(
                semantic_target_question or final_question
            )
            if ref.kind in {"rule", "directive"}
        ]
        if (
            len(structured_exact_refs) == 1
            and answer_focus.startswith((
                "STRUCTURED STATEMENT:",
                "STRUCTURED DETAIL:",
                "STRUCTURED SUMMARY:",
                "STRUCTURED SIMPLE EXPLANATION:",
                "STRUCTURED EXPLANATION WITH CATEGORY:",
                "STRUCTURED EXPLANATION WITH EXAMPLE:",
                "STRUCTURED EXPLANATION:",
            ))
        ):
            precomputed_structured_exact_answer = self._deterministic_structured_answer(
                context=context,
                question=final_question,
                resolved_question=semantic_target_question,
                answer_focus=answer_focus,
            )

        # Narrow conversational MISRA follow-ups that ask only for an already
        # retrieved source rationale or whether a visibly satisfied construct
        # still needs modification are safer and clearer when finalized directly
        # from the accepted Rule/Directive body.  This prevents the model from
        # dropping exact source rationale or reviving an older code snippet.
        precomputed_misra_exact_application = (
            MisraComplianceMode.deterministic_exact_rule_application(
                misra_evidence_question,
                results,
            )
            if (misra_compliance_mode and not LLM_CERTIFICATION_FORCE_GENERATION)
            else ""
        )

        precomputed_misra_scope_guard_answer = (
            MisraComplianceMode.deterministic_scope_guard_answer(
                misra_evidence_question,
                results,
            )
            if (misra_compliance_mode and not LLM_CERTIFICATION_FORCE_GENERATION)
            else ""
        )

        precomputed_misra_yes_no_answer = (
            MisraComplianceMode.deterministic_yes_no_answer(
                misra_evidence_question,
                results,
            )
            if (
                misra_compliance_mode
                and not precomputed_misra_scope_guard_answer
                and not LLM_CERTIFICATION_FORCE_GENERATION
            )
            else ""
        )

        precomputed_misra_semantic_relation_answer = (
            self._semantic_misra_relation_answer(
                misra_evidence_question,
                results,
            )
            if (
                misra_compliance_mode
                and not precomputed_misra_scope_guard_answer
                and not precomputed_misra_yes_no_answer
                and not LLM_CERTIFICATION_FORCE_GENERATION
                and any(
                    isinstance(item, dict) and item.get("_misra_semantic_resolver") is True
                    for item in (results or [])
                )
                and bool(MisraComplianceMode.yes_no_intent(misra_evidence_question))
            )
            else ""
        )

        precomputed_misra_followup_answer = (
            MisraComplianceMode.deterministic_followup_detail(
                final_question,
                results,
            )
            if (
                misra_compliance_mode
                and not precomputed_misra_yes_no_answer
                and not precomputed_misra_semantic_relation_answer
                and not LLM_CERTIFICATION_FORCE_GENERATION
            )
            else ""
        )

        precomputed_misra_reviewer_fast_answer = ""
        misra_reviewer_fast_details = {}
        if (
            misra_compliance_mode
            and not precomputed_misra_guidance_answer
            and not precomputed_misra_requirement_lookup_answer
            and not precomputed_misra_exact_application
            and not precomputed_misra_scope_guard_answer
            and not precomputed_misra_yes_no_answer
            and not precomputed_misra_semantic_relation_answer
            and not precomputed_misra_followup_answer
            and not LLM_CERTIFICATION_FORCE_GENERATION
        ):
            (
                precomputed_misra_reviewer_fast_answer,
                misra_reviewer_fast_details,
            ) = MisraComplianceMode.deterministic_code_review_fast_path(
                misra_evidence_question,
                results,
            )

        misra_llm_generation_requested = (
            misra_compliance_mode
            and not precomputed_misra_guidance_answer
            and not precomputed_misra_requirement_lookup_answer
            and not precomputed_misra_exact_application
            and not precomputed_misra_scope_guard_answer
            and not precomputed_misra_yes_no_answer
            and not precomputed_misra_semantic_relation_answer
            and not precomputed_misra_followup_answer
            and not precomputed_misra_reviewer_fast_answer
            and MisraComplianceMode.should_use_llm_generation(
                misra_evidence_question,
                results,
                grounded_followup=grounded_followup,
            )
        )
        precomputed_misra_answer = (
            precomputed_misra_guidance_answer
            or precomputed_misra_requirement_lookup_answer
            or precomputed_misra_exact_application
            or precomputed_misra_scope_guard_answer
            or precomputed_misra_yes_no_answer
            or precomputed_misra_followup_answer
            or precomputed_misra_reviewer_fast_answer
            or (
            MisraComplianceMode.deterministic_assessment(
                misra_evidence_question,
                results,
            )
            if (
                misra_compliance_mode
                and not LLM_CERTIFICATION_FORCE_GENERATION
                and not misra_llm_generation_requested
            )
            else ""
            )
        )
        if precomputed_misra_yes_no_answer:
            evidence_logger.record_event(
                event_name="MISRA YES/NO FINALIZATION",
                status="SOURCE-GROUNDED",
                details={
                    "intent": MisraComplianceMode.yes_no_intent(final_question),
                    "references": sorted(MisraComplianceMode.available_references(results)),
                    "reason": (
                        "A genuine Yes/No MISRA question had sufficient authoritative "
                        "evidence for deterministic polarity; no LLM rewrite was needed."
                    ),
                },
            )
        if precomputed_misra_reviewer_fast_answer:
            evidence_logger.record_event(
                event_name="MISRA DETERMINISTIC REVIEWER FAST PATH",
                status="SELF-VERIFIED; LLM SKIPPED",
                details=misra_reviewer_fast_details,
            )
        if misra_compliance_mode and misra_llm_generation_requested and not LLM_CERTIFICATION_FORCE_GENERATION:
            evidence_logger.record_event(
                event_name="MISRA GROUNDED LLM GENERATION",
                status="ENABLED",
                details={
                    "model": self.model_router.complex_model,
                    "reason": (
                        "Natural reviewer/explanation, multi-rule, uncertainty, or "
                        "grounded follow-up intent benefits from LLM synthesis while "
                        "retrieval and visible-code assessment remain deterministic."
                    ),
                },
            )
        if misra_compliance_mode and LLM_CERTIFICATION_FORCE_GENERATION:
            evidence_logger.record_event(
                event_name="LLM CERTIFICATION GENERATION GATE",
                status="DETERMINISTIC FINALIZER BYPASSED",
                details={
                    "reason": (
                        "Certification-only process flag requires an actual LLM "
                        "generation over the already-grounded MISRA evidence."
                    ),
                    "production_default": False,
                },
            )
        if precomputed_misra_answer:
            for item in results:
                if isinstance(item, dict):
                    item["_misra_deterministic_assessment"] = True
            evidence_logger.record_event(
                event_name="MISRA DETERMINISTIC FINALIZATION",
                status="SOURCE-GROUNDED",
                details={
                    "references": sorted(MisraComplianceMode.available_references(results)),
                    "reason": (
                        "Strong semantic cues matched actual structured MISRA Rule/Directive bodies; "
                        "finalization did not require an LLM rewrite."
                    ),
                },
            )
        precomputed_identity_answer = self._bm25_identity_fast_path_answer(
            context=context,
            results=results,
            question=final_question,
            resolved_question=semantic_target_question,
            answer_focus=answer_focus,
        )

        if not misra_compliance_mode and not precomputed_structured_exact_answer:
            precomputed_relation_answer = self._grounded_compact_labeled_overview(
                context=context,
                question=final_question,
            )
            if precomputed_relation_answer:
                evidence_logger.record_event(
                    event_name="GROUNDED COMPACT OVERVIEW FINALIZATION",
                    status="DETERMINISTIC",
                    details={
                        "reason": (
                            "A broad overview request matched a compact accepted "
                            "source block with explicit label/value facts; all labels "
                            "were preserved without model omission or invention."
                        )
                    },
                )

        # Same-document approval + eligibility scope follow-up.  Both facts must
        # be explicit in the accepted company context; this never infers an
        # eligible population from the approver role itself.
        if not misra_compliance_mode and not precomputed_relation_answer:
            precomputed_relation_answer = (
                self._grounded_same_document_approval_eligibility_answer(
                    context=context,
                    question=final_question,
                )
            )
            if precomputed_relation_answer:
                evidence_logger.record_event(
                    event_name="GROUNDED SAME-DOCUMENT SCOPE FINALIZATION",
                    status="DETERMINISTIC",
                    details={
                        "reason": (
                            "Approval and eligibility were both explicitly stated "
                            "in the same accepted company context."
                        )
                    },
                )

        if not misra_compliance_mode and not precomputed_relation_answer:
            precomputed_relation_answer = self._grounded_use_eligibility_answer(
                context=context,
                question=final_question,
            )
            if precomputed_relation_answer:
                evidence_logger.record_event(
                    event_name="GROUNDED NATURAL ELIGIBILITY FINALIZATION",
                    status="DETERMINISTIC",
                    details={
                        "reason": (
                            "Natural use/for-whom wording was resolved only because "
                            "accepted company context exposes an explicit eligibility field."
                        )
                    },
                )

        if not misra_compliance_mode and not precomputed_relation_answer:
            precomputed_relation_answer = self._grounded_scope_followup_answer(
                context=context,
                question=final_question,
            )
            if precomputed_relation_answer:
                evidence_logger.record_event(
                    event_name="GROUNDED SCOPE FOLLOW-UP FINALIZATION",
                    status="DETERMINISTIC",
                    details={
                        "reason": (
                            "The follow-up scope term was stated literally in the "
                            "already accepted company context."
                        )
                    },
                )

        # Narrow deterministic relation finalization.  Run this before the
        # model only for an explicitly phrased role/position association where
        # accepted company context contains a direct subject-bound relation.
        # This avoids asking a small model to choose between a person name and
        # the role label already proven by the source.
        if (
            answer_focus.startswith("ENTITY OR CHOICE:")
            and re.search(
                r"(?i)^(?:what|which)\s+.+?\b(?:position|role|title|office)\b"
                r".+?\b(?:associated|connected)\s+with\b",
                final_question,
            )
        ):
            precomputed_relation_answer = self._grounded_direct_relation_answer(
                context=context,
                question=final_question,
                answer_focus=answer_focus,
            )
            if precomputed_relation_answer:
                evidence_logger.record_event(
                    event_name="GROUNDED RELATION FINALIZATION",
                    status="DETERMINISTIC",
                    details={
                        "reason": (
                            "The requested role/position was explicitly bound "
                            "to the named subject in accepted company context."
                        )
                    },
                )

        if (
            not precomputed_relation_answer
            and answer_focus.startswith("APPROVER:")
        ):
            precomputed_relation_answer = self._grounded_approver_answer(
                context=context,
                question=final_question,
                answer_focus=answer_focus,
            )
            if precomputed_relation_answer:
                evidence_logger.record_event(
                    event_name="GROUNDED APPROVER FINALIZATION",
                    status="DETERMINISTIC",
                    details={
                        "reason": (
                            "The approving person/role/entity was explicitly bound "
                            "to the requested approval relation in accepted company context."
                        )
                    },
                )

        if (
            not precomputed_relation_answer
            and answer_focus.startswith("ELIGIBLE OR ENTITLED ENTITY:")
        ):
            precomputed_relation_answer = self._grounded_eligibility_answer(
                context=context,
                question=final_question,
                answer_focus=answer_focus,
            )
            if precomputed_relation_answer:
                evidence_logger.record_event(
                    event_name="GROUNDED ELIGIBILITY FINALIZATION",
                    status="DETERMINISTIC",
                    details={
                        "reason": (
                            "The eligible/entitled population was extracted from "
                            "an explicit eligibility/entitlement source field, not "
                            "from a nearby approval relation."
                        )
                    },
                )

        if (
            not precomputed_relation_answer
            and answer_focus.startswith("RESPONSIBLE ENTITY:")
        ):
            precomputed_relation_answer = (
                self._grounded_labeled_responsible_entity_answer(
                    context=context,
                    question=final_question,
                    answer_focus=answer_focus,
                )
            )
            if precomputed_relation_answer:
                evidence_logger.record_event(
                    event_name="GROUNDED RESPONSIBILITY FINALIZATION",
                    status="DETERMINISTIC",
                    details={
                        "reason": (
                            "An explicit ownership/responsibility label in accepted "
                            "company context was lexically bound to the question; "
                            "the source value was preserved verbatim."
                        )
                    },
                )

        if answer_focus.startswith("COMPOUND:"):
            precomputed_compound_answer = (
                self._grounded_compact_compound_answer(
                    context=context,
                    question=final_question,
                )
            )

        if self._is_explicit_temporal_comparison_question(final_question):
            precomputed_temporal_answer = self._grounded_temporal_comparison_answer(
                context,
                final_question,
            )

            if not precomputed_temporal_answer:
                temporal_retry_started = time.perf_counter()
                (
                    retry_temporal_answer,
                    retry_temporal_context,
                    retry_temporal_results,
                ) = self._grounded_temporal_bm25_retry(
                    context=context,
                    results=results,
                    question=final_question,
                    semantic_target_question=semantic_target_question,
                )
                retrieval_elapsed += time.perf_counter() - temporal_retry_started
                if retry_temporal_answer:
                    precomputed_temporal_answer = retry_temporal_answer
                    context = retry_temporal_context
                    results = retry_temporal_results
                    retrieval_retry_used = True

            if precomputed_temporal_answer:
                evidence_logger.record_event(
                    event_name="TEMPORAL COMPARISON FINALIZATION",
                    status=(
                        "DETERMINISTIC AFTER RETRY"
                        if retrieval_retry_used
                        else "DETERMINISTIC"
                    ),
                    details={
                        "reason": (
                            "Both requested event/date pairs were bound to "
                            "accepted company context before LLM generation."
                        ),
                        "accepted_chunks": len(results),
                    },
                )

        if answer_focus.startswith("COMPARISON:"):
            precomputed_comparison_answer = (
                precomputed_temporal_answer
                or self._grounded_compact_comparison_answer(context, final_question)
            )
        elif precomputed_temporal_answer:
            # Defense-in-depth: even if an upstream normalization variant caused
            # the focus detector to miss the comparison label, a strictly
            # grounded two-event temporal answer must not fall back to a
            # one-date model response.
            precomputed_comparison_answer = precomputed_temporal_answer

        if precomputed_structured_major_family_answer:
            route_decision = ModelRouteDecision(
                route="deterministic",
                model_name=None,
                reason=(
                    "A bare major Rule identifier matched an authoritative child "
                    "Rule family; the user is asked to choose an exact Rule."
                ),
                source_count=len(results),
            )
        elif precomputed_named_section_answer:
            route_decision = ModelRouteDecision(
                route="deterministic",
                model_name=None,
                reason=(
                    "A near-exact structured Section title matched accepted BM25 "
                    "metadata and was rendered directly without vector/reranker/LLM."
                ),
                source_count=len(results),
            )
        elif precomputed_structured_multi_answer:
            route_decision = ModelRouteDecision(
                route="deterministic",
                model_name=None,
                reason="Multiple explicit structured references were explained independently from authoritative anchors.",
                source_count=len(results),
            )
        elif precomputed_structured_comparison_answer:
            route_decision = ModelRouteDecision(
                route="deterministic",
                model_name=None,
                reason="Multiple explicit structured references were compared from authoritative anchors.",
                source_count=len(results),
            )
        elif precomputed_structured_list_answer:
            route_decision = ModelRouteDecision(
                route="deterministic",
                model_name=None,
                reason=(
                    "A complete structured rule-family inventory was retrieved; "
                    "the list is rendered deterministically to prevent omissions."
                ),
                source_count=len(results),
            )
        elif precomputed_structured_exact_answer:
            route_decision = ModelRouteDecision(
                route="deterministic",
                model_name=None,
                reason=(
                    "The exact Rule/Directive request was finalized directly from "
                    "the accepted structured source block."
                ),
                source_count=len(results),
            )
        elif precomputed_misra_semantic_relation_answer:
            route_decision = ModelRouteDecision(
                route="semantic_relation",
                model_name=OLLAMA_FAST_MODEL,
                reason=(
                    "Corpus-derived Rule selection was source-verified and a compact "
                    "structured LLM call resolved only the requested Yes/No relation."
                ),
                source_count=len(results),
            )
        elif precomputed_misra_answer:
            route_decision = ModelRouteDecision(
                route="deterministic",
                model_name=None,
                reason=(
                    "Strong source-proven MISRA Rule/Directive body matches were finalized "
                    "deterministically; ambiguous MISRA assessments still use the complex model."
                ),
                source_count=len(results),
            )
        elif precomputed_identity_answer:
            route_decision = ModelRouteDecision(
                route="deterministic",
                model_name=None,
                reason=(
                    "Strict BM25 identity fast path supplied a direct "
                    "subject-bound source sentence; no model rewrite is needed."
                ),
                source_count=1,
            )
        elif precomputed_relation_answer:
            route_decision = ModelRouteDecision(
                route="deterministic",
                model_name=None,
                reason=(
                    "An explicit grounded relation was extracted deterministically "
                    "from accepted company context."
                ),
                source_count=len(results),
            )
        elif precomputed_comparison_answer:
            route_decision = ModelRouteDecision(
                route="deterministic",
                model_name=None,
                reason="Explicit comparison values/dates were extracted from accepted company context.",
                source_count=len(results),
            )
        elif precomputed_compound_answer:
            route_decision = ModelRouteDecision(
                route="deterministic",
                model_name=None,
                reason=(
                    "Every compact compound facet was matched to an explicit "
                    "value in accepted company context."
                ),
                source_count=len({
                    (
                        item.get("metadata", {}).get("file_path")
                        or item.get("metadata", {}).get("file_name")
                        or "Unknown"
                    )
                    for item in results
                    if isinstance(item, dict)
                }),
            )

            evidence_logger.record_event(
                event_name="COMPOUND GROUNDING SAFETY",
                status="DETERMINISTIC COMPOUND ANSWER",
                details={
                    "reason": (
                        "Every requested compact facet was explicitly matched "
                        "before model generation."
                    )
                },
            )
        else:
            if misra_compliance_mode:
                if self.model_router.forced_model:
                    route_decision = ModelRouteDecision(
                        route="forced",
                        model_name=self.model_router.forced_model,
                        reason="QA forced-model override is active for MISRA assessment.",
                        source_count=len(results),
                    )
                else:
                    route_decision = ModelRouteDecision(
                        route="complex",
                        model_name=self.model_router.complex_model,
                        reason=(
                            "Natural MISRA/code-compliance assessment requires "
                            "grounded multi-rule reasoning by the configured complex model."
                        ),
                        source_count=len(results),
                    )
            else:
                route_decision = self.model_router.choose(
                    question=final_question,
                    resolved_question=semantic_target_question,
                    answer_focus=answer_focus,
                    results=results,
                )

        evidence_logger.record_event(
            event_name="LLM ROUTE",
            status=route_decision.route.upper(),
            details={
                "model": route_decision.model_name or "No LLM",
                "reason": route_decision.reason,
                "source_count": route_decision.source_count,
            },
        )

        if DEBUG_MODE:
            print(
                f"\nFinal Answer Question : "
                f"{final_question}"
            )
            print(
                f"Final Focus Target    : "
                f"{resolved_question}"
            )
            print(
                f"Final Answer Focus    : "
                f"{answer_focus}"
            )
            print(
                f"Final LLM Route       : "
                f"{route_decision.route}"
            )
            print(
                f"Final LLM Model       : "
                f"{route_decision.model_name or 'No LLM'}"
            )

        draft_answer = ""
        verified_answer = ""
        prompt_leak_detected = False
        generation_error = ""
        generation_client = None

        try:
            answer = ""

            if route_decision.route == "deterministic":
                if precomputed_structured_major_family_answer:
                    answer = precomputed_structured_major_family_answer
                elif precomputed_named_section_answer:
                    answer = precomputed_named_section_answer
                elif precomputed_structured_multi_answer:
                    answer = precomputed_structured_multi_answer
                elif precomputed_structured_comparison_answer:
                    answer = precomputed_structured_comparison_answer
                elif precomputed_structured_list_answer:
                    answer = precomputed_structured_list_answer
                elif precomputed_structured_exact_answer:
                    answer = precomputed_structured_exact_answer
                elif precomputed_misra_answer:
                    answer = precomputed_misra_answer
                elif precomputed_identity_answer:
                    answer = precomputed_identity_answer
                elif precomputed_relation_answer:
                    answer = precomputed_relation_answer
                elif precomputed_comparison_answer:
                    answer = precomputed_comparison_answer
                elif precomputed_compound_answer:
                    answer = precomputed_compound_answer
                else:
                    answer = self._deterministic_structured_answer(
                        context=context,
                        question=final_question,
                        resolved_question=semantic_target_question,
                        answer_focus=answer_focus,
                    )

                if answer:
                    draft_answer = answer
                    verified_answer = answer
                    prompt_leak_detected = self._contains_prompt_leak(
                        answer
                    )

                    if precomputed_structured_major_family_answer:
                        generation_skip_reason = (
                            "A bare major Rule identifier was resolved to its grounded "
                            "child Rule family and clarified without model generation."
                        )
                    elif precomputed_named_section_answer:
                        generation_skip_reason = (
                            "Near-exact Section-title evidence was rendered directly "
                            "from accepted structured company context."
                        )
                    elif precomputed_structured_multi_answer:
                        generation_skip_reason = (
                            "Multiple explicit Rule/Directive anchors were explained "
                            "directly from authoritative structured evidence."
                        )
                    elif precomputed_structured_list_answer:
                        generation_skip_reason = (
                            "Complete structured rule-family evidence was rendered "
                            "directly so the model cannot omit retrieved rules."
                        )
                    elif precomputed_structured_exact_answer:
                        generation_skip_reason = (
                            "Exact Rule/Directive evidence was rendered directly "
                            "from the accepted structured source block."
                        )
                    elif precomputed_misra_answer:
                        generation_skip_reason = (
                            "Strong structured MISRA evidence was sufficient for a deterministic, "
                            "reference-grounded assessment."
                        )
                    elif precomputed_identity_answer:
                        generation_skip_reason = (
                            "BM25 identity answer was extracted directly from "
                            "the accepted subject-bound company sentence."
                        )
                    elif precomputed_relation_answer:
                        generation_skip_reason = (
                            "The requested relation/value was extracted directly "
                            "from an explicit subject-bound company relation."
                        )
                    elif precomputed_comparison_answer:
                        generation_skip_reason = (
                            "Comparison answer was extracted deterministically from accepted company context."
                        )
                    elif precomputed_compound_answer:
                        generation_skip_reason = (
                            "Compact compound answer was produced "
                            "deterministically from accepted company context."
                        )
                    else:
                        generation_skip_reason = (
                            "Exact structured answer was produced "
                            "deterministically from accepted company context."
                        )

                    evidence_logger.record_event(
                        event_name="LLM GENERATION",
                        status="SKIPPED",
                        details=generation_skip_reason,
                    )

                else:
                    # Extremely defensive fallback: if a malformed structured
                    # block cannot be extracted deterministically, preserve
                    # v6.2.1 behavior by using a generative model instead of
                    # returning a fabricated or empty answer.
                    fallback_model = (
                        self.model_router.forced_model
                        or self.model_router.fast_model
                    )
                    fallback_route = (
                        "forced"
                        if self.model_router.forced_model
                        else "fast"
                    )
                    route_decision = ModelRouteDecision(
                        route=fallback_route,
                        model_name=fallback_model,
                        reason=(
                            "Deterministic structured extraction was unavailable; "
                            "falling back to grounded generation."
                        ),
                        source_count=route_decision.source_count,
                    )

                    evidence_logger.record_event(
                        event_name="LLM ROUTE",
                        status="DETERMINISTIC FALLBACK",
                        details={
                            "model": fallback_model,
                            "reason": route_decision.reason,
                        },
                    )

            if route_decision.route == "semantic_relation" and precomputed_misra_semantic_relation_answer:
                answer = precomputed_misra_semantic_relation_answer
                draft_answer = answer
                verified_answer = answer
                prompt_leak_detected = self._contains_prompt_leak(answer)
                evidence_logger.record_event(
                    event_name="LLM GENERATION",
                    status="COMPACT SOURCE-GROUNDED RELATION VERDICT",
                    details={
                        "model": OLLAMA_FAST_MODEL,
                        "reason": (
                            "A single structured JSON verdict over one source-verified "
                            "Rule replaced full-context generation."
                        ),
                    },
                )

            if not answer:
                generation_context = context

                if (
                    route_decision.route == "fast"
                    and not GENERATION_USE_FULL_ACCEPTED_CONTEXT
                ):
                    generation_context = self._compact_fast_generation_context(
                        context=context,
                        results=results,
                        question=final_question,
                        resolved_question=semantic_target_question,
                        answer_focus=answer_focus,
                    )

                misra_generation_history = history
                if misra_compliance_mode:
                    isolate_history = MisraComplianceMode.should_isolate_generation_history(
                        final_question,
                        grounded_followup=grounded_followup,
                    )
                    if isolate_history:
                        misra_generation_history = ""
                    evidence_logger.record_event(
                        event_name="MISRA GENERATION HISTORY",
                        status="ISOLATED" if isolate_history else "FOLLOW-UP CONTEXT PRESERVED",
                        details={
                            "grounded_followup": bool(grounded_followup),
                            "current_turn_has_code": MisraComplianceMode.looks_like_c_cpp(final_question),
                            "current_turn_semantic_cues": MisraComplianceMode.semantic_cues(final_question),
                        },
                    )

                prompt = (
                    self._build_misra_compliance_prompt(
                        context=generation_context,
                        question=final_question,
                        history=misra_generation_history,
                        results=results,
                    )
                    if misra_compliance_mode
                    else self._build_prompt(
                        generation_context,
                        final_question,
                        history,
                        answer_focus
                    )
                )

                if DEBUG_MODE:
                    print("\nFinal Prompt:\n")
                    print(prompt)
                    print("=" * 60)

                answer, generation_client = self._generate_for_route(
                    prompt,
                    route_decision,
                )

                evidence_logger.record_event(
                    event_name="LLM MODEL USED",
                    status="GENERATED",
                    details={
                        "route": route_decision.route,
                        "model": generation_client.model_name,
                    },
                )

                answer = self._postprocess_answer(
                    answer,
                    final_question
                )

                if misra_compliance_mode:
                    answer, status_contract_changes = (
                        MisraComplianceMode.normalize_generated_status_contract(
                            misra_evidence_question,
                            answer,
                            results,
                        )
                    )
                    answer, macro_sanitizer_changes = (
                        MisraComplianceMode.sanitize_generated_macro_reconstruction(
                            misra_evidence_question,
                            answer,
                        )
                    )
                    if status_contract_changes:
                        evidence_logger.record_event(
                            event_name="MISRA GENERATION CONTRACT NORMALIZATION",
                            status="APPLIED",
                            details=status_contract_changes,
                        )
                    if macro_sanitizer_changes:
                        evidence_logger.record_event(
                            event_name="MISRA MACRO RECONSTRUCTION GUARD",
                            status="REMOVED NON-LITERAL RECONSTRUCTION",
                            details=macro_sanitizer_changes,
                        )

                draft_answer = answer

                if misra_compliance_mode:
                    # The specialized prompt already enforces the assessment
                    # contract. Generic focus/compound verifiers can erase code
                    # formatting or Rule identifiers, so use deterministic
                    # grounding checks instead.
                    verified_answer = answer
                    prompt_leak_detected = self._contains_prompt_leak(answer)
                    answer = self._apply_output_safety_gate(
                        answer,
                        final_question
                    )
                    if (
                        answer.strip().casefold() != NO_RESULT_MESSAGE.casefold()
                        and not MisraComplianceMode.references_are_grounded(
                            answer, results
                        )
                    ):
                        evidence_logger.record_event(
                            event_name="MISRA REFERENCE GROUNDING GUARD",
                            status="REJECTED",
                            details={
                                "cited": sorted(MisraComplianceMode.cited_references(answer)),
                                "available": sorted(MisraComplianceMode.available_references(results)),
                            },
                        )
                        answer = NO_RESULT_MESSAGE

                    if answer.strip().casefold() != NO_RESULT_MESSAGE.casefold():
                        aligned, alignment_details = (
                            MisraComplianceMode.generated_assessment_is_aligned(
                                misra_evidence_question,
                                answer,
                                results,
                            )
                        )
                        if not aligned:
                            # One guarded LLM self-repair attempt keeps AI synthesis
                            # active for nuanced reviewer questions while refusing to
                            # expose a polarity/code hallucination.  The repair receives
                            # only the same grounded prompt plus machine guard feedback;
                            # it must still pass reference and semantic guards.
                            repair_prompt = (
                                prompt
                                + "\n\nGROUNDING GUARD FEEDBACK FOR REWRITE:\n"
                                + str(alignment_details)
                                + "\nRewrite the answer from scratch. Obey the MISRA "
                                + "EVIDENCE-DERIVED GENERATION CONTRACT exactly. "
                                + "Do not copy any unsupported claim from the rejected draft. "
                                + "Return only the corrected final answer."
                            )
                            try:
                                repaired_answer, repaired_client = self._generate_for_route(
                                    repair_prompt,
                                    route_decision,
                                )
                                generation_client = repaired_client
                                repaired_answer = self._postprocess_answer(
                                    repaired_answer,
                                    final_question,
                                )
                                repaired_answer, repair_status_changes = (
                                    MisraComplianceMode.normalize_generated_status_contract(
                                        misra_evidence_question,
                                        repaired_answer,
                                        results,
                                    )
                                )
                                repaired_answer, repair_macro_changes = (
                                    MisraComplianceMode.sanitize_generated_macro_reconstruction(
                                        misra_evidence_question,
                                        repaired_answer,
                                    )
                                )
                                if repair_status_changes:
                                    evidence_logger.record_event(
                                        event_name="MISRA GENERATION CONTRACT NORMALIZATION",
                                        status="APPLIED DURING REPAIR",
                                        details=repair_status_changes,
                                    )
                                if repair_macro_changes:
                                    evidence_logger.record_event(
                                        event_name="MISRA MACRO RECONSTRUCTION GUARD",
                                        status="REMOVED NON-LITERAL RECONSTRUCTION DURING REPAIR",
                                        details=repair_macro_changes,
                                    )
                                repaired_answer = self._apply_output_safety_gate(
                                    repaired_answer,
                                    final_question,
                                )
                                repaired_grounded = (
                                    repaired_answer.strip().casefold()
                                    != NO_RESULT_MESSAGE.casefold()
                                    and MisraComplianceMode.references_are_grounded(
                                        repaired_answer, results
                                    )
                                )
                                repaired_aligned = False
                                repaired_details = {"reason": "repair_not_grounded"}
                                if repaired_grounded:
                                    repaired_aligned, repaired_details = (
                                        MisraComplianceMode.generated_assessment_is_aligned(
                                            misra_evidence_question,
                                            repaired_answer,
                                            results,
                                        )
                                    )
                                if repaired_grounded and repaired_aligned:
                                    answer = repaired_answer
                                    draft_answer = repaired_answer
                                    verified_answer = repaired_answer
                                    evidence_logger.record_event(
                                        event_name="MISRA SEMANTIC ALIGNMENT GUARD",
                                        status="PASSED AFTER GROUNDED LLM REPAIR",
                                        details={
                                            "initial_rejection": alignment_details,
                                            "repair": repaired_details,
                                        },
                                    )
                                else:
                                    evidence_logger.record_event(
                                        event_name="MISRA SEMANTIC ALIGNMENT GUARD",
                                        status="LLM ANSWER REJECTED AFTER REPAIR; USING GROUNDED FINALIZER",
                                        details={
                                            "initial_rejection": alignment_details,
                                            "repair": repaired_details,
                                        },
                                    )
                                    safe_grounded_answer = (
                                        MisraComplianceMode.deterministic_assessment(
                                            misra_evidence_question,
                                            results,
                                        )
                                    )
                                    answer = safe_grounded_answer or NO_RESULT_MESSAGE
                            except Exception as repair_error:
                                evidence_logger.record_event(
                                    event_name="MISRA SEMANTIC ALIGNMENT GUARD",
                                    status="LLM REPAIR ERROR; USING GROUNDED FINALIZER",
                                    details={
                                        "initial_rejection": alignment_details,
                                        "repair_error": str(repair_error),
                                    },
                                )
                                safe_grounded_answer = (
                                    MisraComplianceMode.deterministic_assessment(
                                        misra_evidence_question,
                                        results,
                                    )
                                )
                                answer = safe_grounded_answer or NO_RESULT_MESSAGE
                        else:
                            evidence_logger.record_event(
                                event_name="MISRA SEMANTIC ALIGNMENT GUARD",
                                status="PASSED",
                                details=alignment_details,
                            )
                else:
                    # Correct cases where a true fact answers the wrong type of
                    # question. Reuse the same routed model so verification does
                    # not unnecessarily swap models on the Ollama server.
                    answer = self._verify_answer_focus(
                        context=context,
                        question=final_question,
                        draft_answer=answer,
                        answer_focus=answer_focus,
                        resolved_question=semantic_target_question,
                        llm_client=generation_client,
                    )

                    answer = self._apply_structured_answer_focus(
                        context=context,
                        question=final_question,
                        resolved_question=semantic_target_question,
                        answer_focus=answer_focus,
                        current_answer=answer
                    )

                    answer = self._apply_grounded_explanation_coverage(
                        context=context,
                        answer_focus=answer_focus,
                        current_answer=answer
                    )

                    if (
                        answer_focus.startswith(
                            "LIST:"
                        )
                        or self._is_multi_answer_question(
                            final_question
                        )
                    ):
                        answer = self._verify_multi_answer(
                            context=context,
                            question=final_question,
                            draft_answer=answer,
                            resolved_question=semantic_target_question,
                            llm_client=generation_client,
                        )

                    if self._is_compound_question(
                        final_question
                    ):
                        answer = self._verify_compound_answer(
                            context=context,
                            question=final_question,
                            draft_answer=answer,
                            resolved_question=semantic_target_question,
                            llm_client=generation_client,
                        )

                    answer = self._relation_aware_date_answer(
                        context=context,
                        question=final_question,
                        resolved_question=semantic_target_question,
                        answer_focus=answer_focus,
                        current_answer=answer
                    )

                    answer = self._remove_invalid_yes_no_prefix(
                        answer,
                        answer_focus
                    )

                    verified_answer = answer

                    # Final deterministic safety gate after every generation and
                    # verification pass.
                    prompt_leak_detected = self._contains_prompt_leak(
                        answer
                    )

                    answer = self._apply_output_safety_gate(
                        answer,
                        final_question
                    )

                    # v6.4.50 technical quality gate: do not let a fluent Qwen
                    # answer introduce unseen numbers, identifiers, proper names
                    # or acronyms.  Ambiguous semantic claims are handled by the
                    # normal grounded prompt/verifiers; high-confidence additions
                    # fail closed instead of triggering another expensive LLM pass.
                    if answer.strip().casefold() != NO_RESULT_MESSAGE.casefold():
                        grounding_check = validate_generated_claims(
                            answer=answer,
                            context=context,
                            question=final_question,
                            results=results,
                        )
                        if not grounding_check.ok:
                            evidence_logger.record_event(
                                event_name="CLAIM GROUNDING GUARD",
                                status="UNSUPPORTED GENERATED CLAIM REJECTED",
                                details={"reasons": list(grounding_check.reasons)},
                            )
                            answer = NO_RESULT_MESSAGE

                if not self._has_substantive_answer_content(answer):
                    recovered = self._recover_compact_source_fact(
                        question=final_question,
                        results=results,
                    )
                    if recovered:
                        evidence_logger.record_event(
                            event_name="ANSWER QUALITY SAFETY",
                            status="RECOVERED FROM ACCEPTED SOURCE",
                            details={
                                "reason": (
                                    "Model output contained no substantive words; "
                                    "a directly supported source sentence was "
                                    "returned instead."
                                )
                            },
                        )
                        answer = recovered
                        verified_answer = recovered
                    else:
                        evidence_logger.record_event(
                            event_name="ANSWER QUALITY SAFETY",
                            status="NON-SUBSTANTIVE OUTPUT REJECTED",
                            details={
                                "reason": (
                                    "Model output contained no substantive words "
                                    "and no safe direct source sentence qualified."
                                )
                            },
                        )
                        answer = NO_RESULT_MESSAGE

        except Exception as error:
            generation_error = str(
                error
            )

            evidence_logger.record_error(
                location="AnswerService.ask",
                error=error,
                details={
                    "question": final_question,
                    "resolved_question": resolved_question,
                    "route": route_decision.route,
                    "model": route_decision.model_name or "No LLM",
                }
            )

            answer = (
                "An error occurred while generating the answer."
            )

        # Presentation is a deterministic, post-safety transformation. It
        # changes layout only and never introduces new factual content.
        if not generation_error:
            answer = self._format_answer_presentation(
                answer=answer,
                question=final_question,
                answer_focus=answer_focus,
            )

        # ======================================
        # DEBUG
        # ======================================
        if DEBUG_MODE:

            print("\nGenerated Answer:\n")
            print(answer)
            print("=" * 60)

        # ======================================
        # FALLBACK RESULT
        # ======================================
        if (
            answer.strip().lower()
            == NO_RESULT_MESSAGE.strip().lower()
        ):

            # Restore the previous valid topic.
            ChatManager.set_current_topic(
                previous_topic
            )

            if DEBUG_MODE:

                print(
                    "\n[TOPIC MEMORY NOT UPDATED] "
                    "Fallback answer returned."
                )

                print(
                    "[FALLBACK SOURCES REMOVED]"
                )

            evidence_logger.record_event(
                event_name="REQUEST LATENCY PROFILE",
                status="GENERATED FALLBACK",
                details={
                    "route": route_decision.route,
                    "model": route_decision.model_name or "No LLM",
                    "retrieval_seconds": round(retrieval_elapsed, 4),
                    "total_seconds": round(
                        time.perf_counter() - request_started,
                        4
                    ),
                },
            )


            evidence_logger.record_answer(
                draft_answer=draft_answer,
                verified_answer=verified_answer,
                final_answer=NO_RESULT_MESSAGE,
                fallback_used=True,
                prompt_leak_detected=prompt_leak_detected,
                sources=[],
                error=generation_error
            )

            self._schedule_post_complex_fast_recovery(generation_client)

            # Important:
            # Do not return unrelated retrieved sources
            # when the answer is not found.
            return {

                "answer": NO_RESULT_MESSAGE,
                "sources": [],
                "chunks": []
            }


        # ======================================
        # VALID ANSWER TOPIC UPDATE
        # ======================================
        canonical_topic = (
            self._extract_topic_from_results(
                results
            )
        )

        if canonical_topic:

            ChatManager.set_current_topic(
                canonical_topic
            )

            if DEBUG_MODE:

                print(
                    f"\n[TOPIC MEMORY UPDATED] "
                    f"{canonical_topic}"
                )


        # Extract sources only for a valid answer. Ordinary answers keep the
        # long-standing single highest-ranked source. A deterministic temporal
        # comparison preserves one accepted source for each grounded date so
        # neither side of a cross-document comparison loses traceability.
        temporal_source_results = []
        if precomputed_temporal_answer:
            temporal_source_results = self._temporal_comparison_source_results(
                results,
                precomputed_temporal_answer,
            )

        deterministic_source_results = []
        if (
            precomputed_relation_answer
            or precomputed_compound_answer
            or precomputed_structured_major_family_answer
            or precomputed_structured_list_answer
            or precomputed_structured_comparison_answer
            or precomputed_structured_multi_answer
            or precomputed_structured_exact_answer
        ):
            deterministic_source_results = self._source_results_supporting_answer(
                results,
                answer,
            )

        if temporal_source_results:
            sources = self._extract_sources(
                temporal_source_results,
                max_sources=None,
            )
        elif (
            precomputed_structured_major_family_answer
            or precomputed_structured_list_answer
            or precomputed_structured_comparison_answer
            or precomputed_structured_multi_answer
            or precomputed_structured_exact_answer
        ):
            sources = self._extract_sources(
                results,
                max_sources=None,
            )
        elif (
            str(answer_focus or "").startswith("REASON:")
            and any(
                isinstance(item, dict) and item.get("_rationale_cross_reference_anchor")
                for item in (results or [])
            )
        ):
            sources = self._extract_sources(
                results,
                max_sources=None,
            )
        elif deterministic_source_results:
            sources = self._extract_sources(
                deterministic_source_results,
                max_sources=None,
            )
        elif misra_compliance_mode:
            sources = self._extract_sources(
                results,
                max_sources=None,
            )
        else:
            sources = self._extract_sources(
                results
            )

        self._store_grounded_followup_state(
            question=final_question,
            context=context,
            results=results,
            misra_compliance_mode=misra_compliance_mode,
            answer_focus=answer_focus,
            previous_state=previous_grounded_state,
            grounded_followup=grounded_followup,
        )

        evidence_logger.record_event(
            event_name="REQUEST LATENCY PROFILE",
            status="COMPLETED",
            details={
                "route": route_decision.route,
                "model": route_decision.model_name or "No LLM",
                "retrieval_seconds": round(retrieval_elapsed, 4),
                "total_seconds": round(
                    time.perf_counter() - request_started,
                    4
                ),
            },
        )

        evidence_logger.record_answer(
            draft_answer=draft_answer,
            verified_answer=verified_answer,
            final_answer=answer,
            fallback_used=False,
            prompt_leak_detected=prompt_leak_detected,
            sources=sources,
            error=generation_error
        )

        self._schedule_post_complex_fast_recovery(generation_client)

        return {

            "answer": answer,

            "sources": sources,

            "chunks": results
        }