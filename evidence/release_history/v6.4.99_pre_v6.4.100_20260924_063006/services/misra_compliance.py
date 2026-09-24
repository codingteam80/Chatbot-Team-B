from __future__ import annotations

import pickle
import re
from typing import Iterable, Mapping, Sequence

from config.settings import BM25_DIR
from utils.structured_reference import extract_structured_references


_REF_RE = re.compile(
    r"\b(?:(?:MISRA\s+)?(?P<kind>Rule|Directive|Dir))\s+"
    r"(?P<identifier>\d+(?:\.\d+)*)\b",
    re.IGNORECASE,
)

_HEADING_REF_RE = re.compile(
    r"(?im)^\s*(?P<kind>Rule|Directive|Dir)\s+"
    r"(?P<identifier>\d+(?:\.\d+)*)\b"
)


class MisraComplianceMode:
    """Intent-first helpers for source-grounded MISRA code assessment.

    The user-facing example questions are deliberately *not* a whitelist.
    Detection combines explicit MISRA/compliance language, conversational
    continuity, and C/C++-like syntax so varied English/Tagalog/mixed prompts
    and pasted code can use the same guarded analysis path.
    """

    _ASSESSMENT_PATTERNS = (
        r"\bcompliant\b",
        r"\bcompliance\b",
        r"\bnon[- ]?compliant\b",
        r"\bviolation(?:s)?\b",
        r"\bviolat(?:e|es|ed|ing)\b",
        r"\bwhich\s+(?:misra\s+)?rule(?:s)?\s+(?:apply|applies|are relevant)\b",
        r"\bwhat\s+(?:misra\s+)?rule(?:s)?\s+(?:apply|applies)\b",
        r"\banong?\s+(?:misra\s+)?rule\b",
        r"\balin(?:g)?\s+(?:misra\s+)?rule\b",
        r"\b(?:is|are)\s+(?:this|that|it)\s+(?:allowed|permitted|valid|acceptable)\b",
        r"\b(?:allowed|permitted)\s+(?:ba|under)\b",
        r"\b(?:valid|okay|ok)\s+(?:ba|lang\s+ba)\b",
        r"\b(?:pwede|puwede)\s+ba\b",
        r"\bmay\s+(?:bang?\s+)?concern\b",
        r"\b(?:check|review|analy[sz]e|assess|evaluate)\b.*\b(?:code|snippet|misra|compliance)\b",
        r"\b(?:i[- ]?check|suriin|tingnan|i[- ]?review|i[- ]?analy[sz]e)\b.*\b(?:code|misra|compliance|violation)\b",
        r"\bpossible\s+(?:misra\s+)?(?:issue|problem|violation)\b",
        r"\b(?:may|meron(?:g)?)\s+(?:bang?\s+)?(?:misra\s+)?(?:issue|violation|problem)\b",
    )

    _CODE_WORD_PATTERNS = (
        r"\bgoto\b",
        r"\b(?:if|for|while|switch)\s*\(",
        r"\b(?:unsigned|signed|const|volatile|static|extern)\b",
        r"\b(?:char|short|int|long|float|double|void|bool|size_t|uint\d+_t|int\d+_t)\b",
        r"\b(?:struct|union|enum|typedef)\b",
        r"\b(?:malloc|calloc|realloc|free|sizeof)\s*\(",
        r"\breturn\b\s+[^\n;]+;",
        r"#\s*(?:include|define|undef|if|ifdef|ifndef|pragma)\b",
        r"->|\+\+|--|&&|\|\||<<|>>|==|!=|<=|>=",
    )

    _EXACT_LOOKUP_ONLY = re.compile(
        r"^\s*(?:(?:what|ano)\s+(?:does|ang)\s+)?"
        r"(?:(?:misra\s+)?(?:rule|directive|dir))\s+"
        r"\d+(?:\.\d+)*"
        r"(?:\s+(?:say|says|mean|means))?\s*[?.!]*\s*$",
        re.IGNORECASE,
    )

    @classmethod
    def looks_like_c_cpp(cls, text: str) -> bool:
        value = str(text or "")
        if not value.strip():
            return False

        # A preprocessor directive is already a strong C/C++ syntax signal,
        # even when it appears after a short natural-language lead-in such as
        # ``Paano naman kung ganito? #define ...``.  Treat the directive token
        # itself as authoritative code syntax rather than requiring it at column 1.
        if re.search(r"#\s*(?:include|define|undef|if|elif|ifdef|ifndef|endif|pragma)\b", value, re.IGNORECASE):
            return True

        score = 0
        if ";" in value:
            score += 1
        if "{" in value and "}" in value:
            score += 1
        if "\n" in value and (";" in value or "{" in value):
            score += 1
        score += sum(
            1 for pattern in cls._CODE_WORD_PATTERNS
            if re.search(pattern, value, re.IGNORECASE)
        )
        return score >= 2

    @classmethod
    def informational_reference_query(cls, question: str) -> bool:
        """Return True for source-information requests about explicit MISRA IDs.

        These questions ask what a Rule/Directive says, means, is classified as,
        why it exists, or what source examples it provides.  They are not code
        compliance assessments and should use the exact structured-document path.
        """

        raw = str(question or "").strip()
        if not raw or cls.looks_like_c_cpp(raw):
            return False
        if not _REF_RE.search(raw) and not re.search(
            r"\b(?:rules?|directives?|dirs?)\s+\d+(?:\.\d+)*",
            raw,
            re.IGNORECASE,
        ):
            return False

        clean = re.sub(r"\s+", " ", raw).casefold()
        if re.search(
            r"\b(?:this|that|shown|above|following)\s+(?:code|snippet|function|expression|statement|case)\b",
            clean,
        ):
            return False

        # Asking the source for compliant/non-compliant *examples* is still a
        # document-information request; the adjectives describe the requested
        # example labels, not the user's own code status.
        asks_source_example = bool(re.search(
            r"\b(?:example|examples|sample|illustrat(?:e|ion))\b",
            clean,
        ))
        if (
            not asks_source_example
            and re.search(
                r"\b(?:compliant|compliance|non[- ]?compliant|violation|violates?|review|assess|evaluate|check\s+(?:this|the)\s+code)\b",
                clean,
            )
        ):
            return False

        return bool(re.search(
            r"\b(?:state|states|say|says|mean|means|meaning|explain|describe|discuss|"
            r"summari[sz]e|summary|tell\s+me\s+about|why|reason|rationale|"
            r"example|examples|sample|illustrat(?:e|ion)|category|classification|"
            r"mandatory|required|advisory|compare|comparison|contrast|difference)\b",
            clean,
        ))

    @classmethod
    def informational_catalog_query(cls, question: str) -> bool:
        """Return True for broad MISRA inventory/list questions, not code review.

        Examples include ``List mandatory MISRA rules`` and ``Which MISRA rules
        apply to pointers?``.  The structured retriever can answer those from the
        authoritative Rule inventory without forcing compliance-assessment mode.
        """

        raw = str(question or "").strip()
        if not raw or cls.looks_like_c_cpp(raw):
            return False
        clean = re.sub(r"\s+", " ", raw).casefold()
        if not re.search(r"\b(?:rule|rules|directive|directives|guideline|guidelines)\b", clean):
            return False
        if re.search(
            r"\b(?:this|that|shown|above|following|my)\s+(?:code|snippet|function|expression|statement|case)\b|\bhere\b",
            clean,
        ):
            return False
        return bool(
            re.match(r"^(?:list|enumerate|name|what\s+are|which|what\s+rules?|which\s+rules?)\b", clean)
            or re.search(r"\b(?:mandatory|required|advisory)\s+(?:misra\s+)?rules?\b", clean)
            or re.search(r"\brules?\s+(?:for|related\s+to|about|applicable\s+to)\b", clean)
        )

    @classmethod
    def natural_requirement_lookup_intent(cls, question: str) -> bool:
        """Return True for text-only requests naming a MISRA concept itself.

        Examples include short noun phrases such as "unused parameter" or
        "pointer from the Standard Library".  These ask which requirement
        applies, not whether visible user code is compliant.
        """

        raw = str(question or "").strip()
        if not raw or cls.looks_like_c_cpp(raw):
            return False
        if not cls.semantic_cues(raw):
            return False
        if cls.yes_no_intent(raw):
            return False
        clean = re.sub(r"\s+", " ", raw.casefold())
        if re.search(r"\b(?:why|bakit|reason|rationale|purpose|dahilan|layunin)\b", clean):
            return False
        if re.search(
            r"\b(?:review|assess|evaluate|check\s+(?:this|the)\s+code|compliant|"
            r"compliance|non[- ]?compliant|violation|violates?)\b",
            clean,
        ):
            return False
        return True

    @classmethod
    def build_rerank_query(cls, question: str, resolved_question: str = "") -> str:
        """Build a source-language reranker query for multilingual MISRA turns.

        Retrieval still uses the full original/search question.  Only the BGE
        relevance comparison is canonicalized when deterministic source-language
        semantic cues are available, preventing Tagalog surface wording from
        demoting an exact English Rule body that recall already found.
        """

        raw = str(question or "").strip()
        cues = cls.semantic_cues(raw)
        if cues:
            return "MISRA C | " + " | ".join(cues)
        resolved = str(resolved_question or "").strip()
        return resolved or raw

    @classmethod
    def post_normalization_rescue_query(
        cls,
        original_question: str,
        normalized_query: str,
        current_topic: str = "",
    ) -> str:
        """Return a normalized MISRA query only when deterministic rescue is safe.

        General multilingual normalization remains untouched.  Promotion is
        allowed only when the user explicitly mentioned MISRA (or the confirmed
        current topic is MISRA) *and* the normalized query exposes a known
        source-language requirement cue.  The returned query can then be sent
        through the authoritative Rule-body rescue before MultiQuery/reranking.
        """

        original = re.sub(r"\s+", " ", str(original_question or "")).strip()
        normalized = re.sub(r"\s+", " ", str(normalized_query or "")).strip()
        topic = re.sub(r"\s+", " ", str(current_topic or "")).strip()
        if not normalized:
            return ""
        if "misra" not in original.casefold() and "misra" not in topic.casefold():
            return ""
        if not cls.semantic_cues(normalized):
            return ""
        if not cls.is_request(normalized, current_topic=topic or "MISRA"):
            return ""
        return normalized

    @classmethod
    def guidance_intent(cls, question: str) -> bool:
        """Return True for prospective/how-to MISRA guidance, not code verdicts.

        A user can mention words such as ``non-compliant`` while asking how to
        *avoid* a future violation.  That is materially different from asking
        whether visible code already violates MISRA.  Keep this classifier
        wording-driven and identifier-free so paraphrases, Taglish, and ordinary
        reviewer language use the same path without matching one canned question.
        """

        raw = str(question or "").strip()
        if not raw or cls.looks_like_c_cpp(raw):
            return False
        if cls.informational_catalog_query(raw):
            return False

        clean = re.sub(r"\s+", " ", raw).casefold()

        # Direct verdict requests keep assessment semantics.  However, prospective
        # wording such as "What should I check before converting ..." is guidance,
        # not a review verdict merely because it contains the verb "check".
        if cls.yes_no_intent(raw):
            return False

        rule_scope = bool(re.search(
            r"\b(?:misra|rules?|directives?|guidelines?|requirements?)\b",
            clean,
        ))
        if not rule_scope:
            return False

        prospective = bool(re.search(
            r"\b(?:plan(?:ning)?|intend|want|gusto|balak|before|bago|"
            r"convert|converting|refactor|refactoring|gawin|gawing|papalitan|"
            r"kailangan|need|should|watch|bantayan|consider|keep\s+in\s+mind|"
            r"applicable|relevant|apply|applies)\b",
            clean,
        ))
        avoidance = bool(re.search(
            r"\b(?:avoid|prevent|para\s+(?:hindi|di)|so\s+(?:it|this)\s+doesn'?t|"
            r"not\s+become|huwag|wag)\b.{0,45}"
            r"\b(?:non[- ]?compliant|violation|issue|problem)\b",
            clean,
        ))
        broad_rule_request = bool(re.search(
            r"\b(?:what|which|ano|anong|alin(?:g)?)\b.{0,80}"
            r"\b(?:rules?|directives?|guidelines?|requirements?)\b",
            clean,
        ))
        broad_review_guidance = bool(re.search(
            r"\b(?:common|typical|key|main)\b.{0,40}"
            r"\b(?:mistakes?|issues?|checks?|concerns?|pitfalls?)\b",
            clean,
        ))

        if prospective or avoidance or broad_rule_request or broad_review_guidance:
            return True

        # A true present-tense review/check request remains an assessment.  Keep
        # this after prospective detection so "what should I check before ..."
        # cannot be misclassified as an existing violation review.
        if re.search(
            r"\b(?:review|check|analy[sz]e|assess|evaluate|verify|"
            r"i[- ]?review|i[- ]?check|suriin|tingnan)\b",
            clean,
        ):
            return False
        return False

    @classmethod
    def is_request(cls, question: str, current_topic: str = "") -> bool:
        raw = str(question or "").strip()
        if not raw:
            return False

        # Preserve source-information and inventory questions on the exact
        # structured-document path.  They ask about the standard itself, not
        # whether user code is compliant.
        if cls._EXACT_LOOKUP_ONLY.match(raw):
            return False
        if cls.informational_reference_query(raw):
            return False
        if cls.informational_catalog_query(raw):
            return False

        lowered = raw.casefold()
        topic = str(current_topic or "").casefold()
        explicit_misra = "misra" in lowered
        misra_context = "misra" in topic
        code_like = cls.looks_like_c_cpp(raw)
        natural_requirement = bool(cls._natural_requirement_cues(raw))
        assessment = any(
            re.search(pattern, raw, re.IGNORECASE)
            for pattern in cls._ASSESSMENT_PATTERNS
        )
        explicit_reference = bool(_REF_RE.search(raw))
        explicit_review = bool(re.search(
            r"\b(?:review|check|analy[sz]e|assess|evaluate|verify)\b",
            raw,
            re.IGNORECASE,
        ))

        # An explicit Rule/Directive identifier plus review/assessment wording
        # is a MISRA application request even when the user omits the word
        # "MISRA" and does not paste literal C syntax. Exact lookup-only
        # questions were already excluded above.
        if explicit_reference and (
            assessment
            or explicit_review
            or bool(cls.yes_no_intent(raw))
            or natural_requirement
            or bool(re.search(
                r"\b(?:explain|meaning|mean|simple\s+terms?|plain\s+english|"
                r"why|rationale|reason|example|examples|sample|illustrat|apply|applies)\b",
                lowered,
            ))
        ):
            return True

        # Natural source-backed requirement questions can also be direct
        # yes/no/permission questions without saying "MISRA" (for example,
        # "Is it ok to use trigraphs?"). Once a known source-language cue is
        # present, use the guarded MISRA path rather than generic generation.
        if natural_requirement and cls.yes_no_intent(raw):
            return True

        # Natural English prose can ask for rationale/application without a Rule
        # number or the word MISRA in the same sentence.  Once a source-language
        # requirement cue is independently recognized, these relation words are
        # sufficient to use the guarded MISRA path.
        if natural_requirement and re.search(
            r"\b(?:why|rationale|reason|discouraged|satisfy|satisfies|satisfied|"
            r"minimum|requirement|what\s+if|what\s+about|example|examples)\b",
            lowered,
        ):
            return True

        # Explicit MISRA + assessment/code intent is the strongest signal.
        # A recognized natural programming concept also qualifies even when the
        # user provides no code (for example a text-only missing-return-path
        # scenario).
        if explicit_misra and (assessment or code_like or natural_requirement):
            return True

        # Follow-ups such as "Which rule applies here?" inherit a confirmed
        # MISRA topic. Natural programming concepts (recursion, dynamic memory,
        # etc.) are also sufficient because their source-language cues can be
        # matched directly against the authoritative Rule/Directive inventory.
        if misra_context and (assessment or code_like or natural_requirement):
            return True

        # Natural Rule/Directive questions may omit the word "MISRA" while
        # still being explicit about the standard-reference relationship, e.g.
        # "What rule covers recursion?". Keep this identifier-free: the
        # concept cue is matched to the authoritative corpus later.
        if natural_requirement and re.search(
            r"\b(?:rule|rules|directive|directives|guideline|guidelines)\b",
            lowered,
        ):
            return True

        # MISRA-first behavior: a pasted C/C++ snippet on its own is treated as
        # a request for compliance analysis. Ordinary prose is not.
        if code_like:
            return True

        # Natural code scenario without syntax: require both assessment wording
        # and a C/C++-specific signal so unrelated company-policy questions stay
        # on the existing general path.
        code_signal = any(
            re.search(pattern, raw, re.IGNORECASE)
            for pattern in cls._CODE_WORD_PATTERNS
        )
        return assessment and code_signal

    @classmethod
    def yes_no_intent(cls, question: str) -> str:
        """Return the conservative YES/NO relation requested by the user.

        This is intentionally wording-driven and identifier-free.  It does not
        decide the MISRA answer by itself; it only tells the deterministic
        finalizer whether the user is asking about permission, a requirement,
        violation status, or compliance status.  Polite requests such as
        ``Can you explain ...`` are deliberately excluded.
        """

        clean = re.sub(r"\s+", " ", str(question or "").strip().casefold())
        if not clean:
            return ""

        if re.match(r"^(?:can|could|would|will)\s+you\b", clean):
            return ""

        # A category/classification choice is not a binary requirement claim.
        # ``Is Rule 14.3 required, mandatory, or advisory?`` asks for the
        # source Category field and must never be converted into Yes/No.
        if (
            re.search(r"\b(?:rule|directive|dir)\s+\d+(?:\.\d+)*\b", clean)
            and re.search(r"\b(?:mandatory|required|advisory)\b", clean)
            and (
                re.search(r"\bcategory|classification|classified\b", clean)
                or len(re.findall(r"\b(?:mandatory|required|advisory)\b", clean)) >= 2
            )
        ):
            return ""

        # Direct compliance/non-compliance polarity.
        if (
            re.search(r"\b(?:compliant|compliance|non[- ]?compliant|non[- ]?compliance)\b", clean)
            and (
                re.match(r"^(?:is|are|was|were|does|do|did|can|could|should|must|has|have|had)\b", clean)
                or re.search(r"\b(?:ba|baga)\b", clean)
            )
        ):
            return "compliance"

        # Explicit question about whether the shown/described behavior violates
        # a MISRA requirement.
        if (
            re.search(r"\b(?:violate|violates|violated|violation|non[- ]?compliance)\b", clean)
            and (
                re.match(r"^(?:is|are|was|were|does|do|did|can|could|should|must|has|have|had)\b", clean)
                or re.search(r"\b(?:may|meron(?:g)?)\b.{0,18}\b(?:violation|issue|problem)\b", clean)
            )
        ):
            return "violation"

        # Permission/allowability wording.  ``Can I use malloc ...?`` and
        # natural Tagalog ``Pwede ba ...?`` belong here, while ``Can you ...``
        # was excluded above as a polite imperative.
        if (
            re.match(r"^(?:does|do|did|is|are|was|were|can|could|should|may)\b", clean)
            and re.search(r"\b(?:allow|allows|allowed|permit|permits|permitted|acceptable|valid|okay|ok|use|used|using|have|leave|keep|omit|omits|omitted|omitting|exclude|excluded|skip|skipped|remove|removed)\b", clean)
        ) or re.match(r"^(?:pwede|puwede)\s+ba\b", clean):
            return "permission"

        # Requirement/prohibition wording.  The finalizer compares the
        # proposition polarity with the actual source statement so questions
        # such as ``Does MISRA require recursion?`` cannot be answered Yes when
        # the source actually says recursion is prohibited.
        if (
            re.match(r"^(?:does|do|did|is|are|was|were|must|should|has|have|had)\b", clean)
            and re.search(
                r"\b(?:require|requires|required|prohibit|prohibits|prohibited|"
                r"forbid|forbids|forbidden|restrict|restricts|restricted|"
                r"disallow|disallows|disallowed|must|shall|should)\b",
                clean,
            )
        ) or re.search(r"^(?:bawal|kailangan|required)\s+ba\b", clean):
            return "requirement"

        return ""

    @staticmethod
    def _statement_is_prohibitive(statement: str) -> bool:
        clean = re.sub(r"\s+", " ", str(statement or "").strip().casefold())
        if not clean:
            return False
        return bool(re.search(
            r"\b(?:shall|should|must|may)\s+not\b|"
            r"\bthere\s+(?:shall|should|must)\s+be\s+no\b|"
            r"\b(?:shall|should|must)\s+be\s+no\b|"
            r"\b(?:prohibited|forbidden|disallowed)\b|"
            r"\bshall\s+never\b|\bshall\s+only\b.*\bnot\b",
            clean,
        ))

    @staticmethod
    def _question_claim_is_prohibitive(question: str) -> bool:
        clean = re.sub(r"\s+", " ", str(question or "").strip().casefold())
        return bool(re.search(
            r"\b(?:not|never|prohibit|prohibits|prohibited|forbid|forbids|"
            r"forbidden|restrict|restricts|restricted|disallow|disallows|"
            r"disallowed|bawal|hindi\s+(?:allowed|pinapayagan|pwede|puwede))\b",
            clean,
        ))

    @classmethod
    def deterministic_yes_no_answer(
        cls,
        question: str,
        results: Sequence[Mapping],
    ) -> str:
        """Return a source-grounded direct YES/NO answer when determinable.

        The method runs only after authoritative MISRA retrieval has attached a
        source-language cue and deterministic assessment state.  It never turns
        uncertainty into a guessed Yes/No and never makes a whole-program
        compliance claim.
        """

        intent = cls.yes_no_intent(question)
        if not intent:
            return ""

        candidates = [
            item for item in (results or [])
            if isinstance(item, Mapping)
            and (
                item.get("_misra_rule_body_rescue") is True
                or bool(item.get("_structured_topic_family"))
            )
            and cls._is_citable_rule_body_record(item)
        ]
        if not candidates or not cls.supports_requested_standard(question, candidates):
            return ""

        for item in candidates:
            if (
                item.get("_misra_rule_body_rescue") is True
                and float(item.get("_misra_cue_coverage", 0.0) or 0.0) < 0.72
            ):
                return ""

        states = [
            str(item.get("_misra_assessment_state", "violation") or "violation").casefold()
            for item in candidates
        ]

        # Uncertainty about a *specific code instance* must never be converted
        # into a guessed Yes/No.  A text-only requirement/permission question is
        # different: the Rule/Directive statement itself can directly establish
        # whether the standard requires or prohibits the described behavior.
        uncertain_present = any(state == "uncertain" for state in states)
        source_proposition_question = (
            intent == "requirement"
            or (intent == "permission" and not cls.looks_like_c_cpp(question))
        )
        if uncertain_present and not source_proposition_question:
            observations = [
                re.sub(r"\s+", " ", str(item.get("_misra_observation", "") or "")).strip()
                for item in candidates
            ]
            observation = next((value for value in observations if value), "")
            if observation:
                return f"Needs more context. {observation}"
            return "Needs more context. The accepted MISRA evidence does not establish a definite Yes or No for the described case."

        statements = [cls._source_rule_statement(str(item.get("text", "") or "")) for item in candidates]
        source_negative = [cls._statement_is_prohibitive(statement) for statement in statements]
        support_statements = list(statements)

        # Generic omission/absence permission questions can be decided when
        # the accepted source explicitly requires the named thing to be present
        # (for example, asking whether a required label may be omitted).
        absence_request = bool(re.search(
            r"\b(?:omit|omits|omitted|omitting|without|leave\s+out|left\s+out|"
            r"exclude|excluded|missing|absent)\b",
            re.sub(r"\s+", " ", str(question or "").casefold()),
        ))
        required_presence_statement = ""
        if intent == "permission" and absence_request:
            question_tokens = set(cls._lexical_tokens(question))
            for statement in statements:
                normalized_statement = re.sub(r"\s+", " ", str(statement or "")).strip()
                if not re.search(
                    r"(?i)\b(?:shall|must)\b.{0,100}\b(?:have|contain|include|provide|be\s+present)\b",
                    normalized_statement,
                ):
                    continue
                statement_tokens = set(cls._lexical_tokens(normalized_statement))
                if len(question_tokens.intersection(statement_tokens)) >= 1:
                    required_presence_statement = normalized_statement
                    break

        if intent == "permission" and absence_request and required_presence_statement:
            reference = ""
            for item in candidates:
                metadata = item.get("metadata", {}) or {}
                identifier = str(metadata.get("directive_id", "") or metadata.get("rule_id", "") or "").strip()
                if identifier:
                    kind = "Directive" if str(metadata.get("section_type", "rule") or "rule").casefold() == "directive" else "Rule"
                    reference = f"{kind} {identifier}"
                    break
            sentence = required_presence_statement.rstrip(" .")
            if sentence:
                sentence = sentence[:1].lower() + sentence[1:]
            answer = f"No. {reference + ' states that ' if reference else ''}{sentence}.".strip()
            return answer if cls.references_are_grounded(answer, candidates) else ""

        # Some permission questions ask whether a required body may be empty.
        # The decisive source language can live in an attached Amplification
        # rather than in the short Rule title.  Detect only explicit absence
        # wording and an explicit positive source requirement (shall/must
        # contain/have/include) so this cannot generalize into guesswork.
        empty_request = bool(re.search(
            r"\b(?:empty|blank|without|nothing|no\s+action)\b",
            re.sub(r"\s+", " ", str(question or "").casefold()),
        ))
        required_presence = False
        if intent == "permission" and empty_request:
            for index, item in enumerate(candidates):
                source_text = re.sub(r"\s+", " ", str(item.get("text", "") or "")).strip()
                match = re.search(
                    r"(?i)([^.!?]{0,120}\b(?:shall|must)\b[^.!?]{0,120}"
                    r"\b(?:contain|have|include|provide)\b[^.!?]{0,160}[.!?])",
                    source_text,
                )
                if match:
                    sentence = re.sub(r"\s+", " ", match.group(1)).strip()
                    question_tokens = set(cls._lexical_tokens(question))
                    sentence_tokens = set(cls._lexical_tokens(sentence))
                    if (
                        len(question_tokens.intersection(sentence_tokens)) >= 1
                        or ("else" in str(question).casefold() and "else" in sentence.casefold())
                    ):
                        required_presence = True
                        support_statements[index] = sentence

        if intent == "permission" and empty_request and required_presence:
            # A natural question such as ``Can I have an empty else block?``
            # is underspecified: Rule 15.7 constrains the terminating else of
            # an if/else-if chain, not every possible simple-if shape.  Give
            # the exact grounded condition instead of an over-broad Yes/No.
            for item, statement in zip(candidates, support_statements):
                source_text = re.sub(r"\s+", " ", str(item.get("text", "") or "")).strip()
                if "else" not in source_text.casefold():
                    continue
                if re.search(r"(?i)shall\s+contain\s+at\s+least\s+either\s+one\s+side\s+effect\s+or\s+a\s+comment", source_text):
                    return (
                        "No. For the terminating `else` of an `if` / `else if` chain, "
                        "an empty block by itself is not sufficient. Rule 15.7 says "
                        "the `else` must contain at least one side effect or a comment. "
                        "This answer is scoped to the terminating `else` requirement "
                        "established by the accepted MISRA evidence."
                    )

        if intent == "compliance":
            yes = all(state == "compliant" for state in states)
        elif intent == "violation":
            yes = any(state == "violation" for state in states)
        elif intent == "permission":
            # For visible code use the deterministic assessment state. For a
            # text-only concept question, the source statement itself expresses
            # whether the described behavior is allowed or prohibited.
            if cls.looks_like_c_cpp(question):
                yes = all(state == "compliant" for state in states)
            elif required_presence:
                yes = False
            else:
                yes = not any(source_negative)
        else:  # requirement / prohibition proposition
            question_negative = cls._question_claim_is_prohibitive(question)
            # When several source requirements apply, require them to agree in
            # proposition polarity before emitting a binary answer.
            if len(set(source_negative)) > 1:
                return ""
            yes = bool(source_negative[0]) == bool(question_negative)

        lead = "Yes." if yes else "No."
        support: list[str] = []
        for item, statement in zip(candidates, support_statements):
            metadata = item.get("metadata", {}) or {}
            section_type = str(metadata.get("section_type", "rule") or "rule").casefold()
            identifier = str(
                metadata.get("directive_id", "")
                or metadata.get("rule_id", "")
                or ""
            ).strip()
            kind = "Directive" if section_type == "directive" else "Rule"
            reference = f"{kind} {identifier}".strip()
            if statement:
                sentence = statement.rstrip(" .")
                if sentence:
                    sentence = sentence[:1].lower() + sentence[1:]
                    support.append(f"{reference} states that {sentence}.")
            elif reference:
                support.append(f"{reference} is the matched source requirement.")

        answer = " ".join([lead, *support]).strip()
        return answer if cls.references_are_grounded(answer, candidates) else ""

    @staticmethod
    def _code_only_view(question: str) -> str:
        """Extract conservative C-like lines from a mixed natural-language prompt.

        The full original question is still preserved for intent/retrieval. This
        view is used only by syntax/data-flow interpretation so phrases such as
        ``Okay ba yung if(p)?`` cannot be mistaken for executable code.
        """
        raw = str(question or "").replace("\r\n", "\n").replace("\r", "\n")
        lines = raw.split("\n")
        kept: list[str] = []
        code_started = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if code_started:
                    kept.append("")
                continue
            inline_directive = re.search(
                r"#\s*(?:include|define|undef|if|elif|ifdef|ifndef|endif|pragma)\b",
                stripped,
                re.I,
            )
            if inline_directive:
                code_started = True
                kept.append(stripped[inline_directive.start():])
                continue

            looks = bool(
                re.search(r"[;{}]", stripped)
                or re.match(r"^#\s*(?:include|define|undef|if|elif|ifdef|ifndef|endif|pragma)\b", stripped, re.I)
                or re.match(r"^(?:if|else\s+if|else|for|while|switch|case\b|default\s*:|return\b|goto\b|break\b|continue\b)", stripped, re.I)
                or re.match(
                    r"^(?:(?:const|volatile|static|extern|signed|unsigned|register|auto|typedef)\s+)*"
                    r"(?:void|char|short|int|long|float|double|bool|_Bool|size_t|u?int\d+_t|intptr_t|uintptr_t|struct\s+\w+|union\s+\w+|enum\s+\w+)\b",
                    stripped,
                    re.I,
                )
                or re.match(r"^[A-Za-z_]\w*\s*(?:\[[^]]+\])?\s*(?:=|\+=|-=|\+\+|--)", stripped)
                or re.match(r"^[A-Za-z_]\w*\s*\([^)]*\)\s*;?$", stripped)
            )
            # A question sentence may contain code-looking tokens. A literal
            # question mark is a strong prose signal unless the same line also
            # contains a statement terminator/preprocessor directive.
            if "?" in stripped and ";" not in stripped and not stripped.startswith("#"):
                looks = False
            if looks:
                code_started = True
                kept.append(line)
        return "\n".join(kept).strip()

    @classmethod
    def _natural_requirement_cues(cls, question: str) -> list[str]:
        """Return source-language requirement cues from explicit human concepts.

        This layer is intentionally identifier-free.  It recognizes ordinary
        English/Tagalog descriptions of a programming concern and emits wording
        that can be matched against the authoritative MISRA Rule/Directive body
        inventory.  It is used mainly when the user asks a natural MISRA question
        without pasting enough code for the syntax interpreter to decide the
        construct directly.
        """
        raw = str(question or "")
        clean = re.sub(r"\s+", " ", raw).casefold()
        output: list[str] = []

        def add(value: str) -> None:
            if value and value not in output:
                output.append(value)

        # Missing return on one execution/exit path of a value-returning function.
        if (
            re.search(r"\b(?:execution|exit|control)\s+paths?\b", clean)
            and re.search(r"\b(?:no|without|missing|walang)\b.{0,28}\breturn(?:ed)?(?:\s+value)?\b", clean)
        ) or re.search(r"\b(?:path|daan)\b.{0,30}\bwalang\s+return\b", clean):
            add("all exit paths from a function with non- void return type shall have an explicit return statement with an expression")

        if re.search(r"\b(?:uninitiali[sz]ed|uninitialised|before\s+(?:it\s+)?(?:is\s+)?set|bago\s+ma[- ]?set|hindi\s+na[- ]?set)\b", clean):
            add("the value of an object with automatic storage duration shall not be read before it has been set")

        if re.search(r"\b(?:curly\s+braces?|braces?|compound[- ]statement)\b", clean):
            add("the body of an iteration-statement or a selection-statement shall be a compound-statement")

        if re.search(r"\bpointer\s+(?:arithmetic|manipulation)\b", clean):
            add("the +, -, += and -= operators should not be applied to an expression of pointer type")

        if re.search(r"\b(?:array\s+(?:bound|bounds|index|subscript)|out[- ]of[- ]bounds?|invalid\s+array\s+(?:index|subscript))\b", clean):
            add("a pointer resulting from arithmetic on a pointer operand shall address an element of the same array as that pointer operand")

        if (
            re.search(r"\bsigned(?:ness)?\b", clean)
            and re.search(r"\bunsigned\b", clean)
            and re.search(r"\b(?:compare|comparison|operator|operand|type)\b", clean)
        ):
            add("both operands of an operator in which the usual arithmetic conversions are performed shall have the same essential type category")

        if re.search(r"\b(?:macro\s+(?:parameters?|expansion)|parameter\s+parentheses|macro.*parenthes)\b", clean):
            add("expressions resulting from the expansion of macro parameters shall be enclosed in parentheses")

        # Character-set question: keep this identifier-free and match the
        # authoritative Rule body by its own wording. This prevents a nearby
        # Rationale sentence about permitted *digraphs* from being mistaken as
        # permission to use trigraphs.
        if re.search(r"\btrigraphs?\b", clean):
            add("trigraphs should not be used")

        # Unused function parameters.  Support ordinary English, Tagalog, and
        # Taglish paraphrases *before* MultiQuery/reranking.  Reviewers often
        # say "function argument" colloquially when they mean a declared
        # parameter, so accept that wording only when an explicit unused / not
        # used concept is present.  This keeps the mapping narrow while making
        # semantically equivalent phrasing converge on the same source Rule.
        unused_parameter_term = r"(?:function\s+)?(?:parameters?|arguments?)"
        unused_parameter_concept = bool(
            re.search(
                rf"\bunused\b.{{0,36}}\b{unused_parameter_term}\b|"
                rf"\b{unused_parameter_term}\b.{{0,36}}\bunused\b",
                clean,
            )
            or re.search(
                rf"\b{unused_parameter_term}\b.{{0,42}}\b(?:never|not)\s+(?:be\s+)?used\b",
                clean,
            )
            or re.search(
                rf"\b(?:never|not)\s+(?:be\s+)?used\b.{{0,42}}\b{unused_parameter_term}\b",
                clean,
            )
            or re.search(
                rf"\b(?:hindi|di)\s+(?:ginagamit|nagagamit|used)\b.{{0,36}}\b{unused_parameter_term}\b",
                clean,
            )
            or re.search(
                rf"\b{unused_parameter_term}\b.{{0,36}}\b(?:hindi|di)\s+(?:ginagamit|nagagamit|used)\b",
                clean,
            )
        )
        if unused_parameter_concept:
            add("there should be no unused parameters in functions")

        # Goto-target label scope.  Keep the cue equal to the authoritative
        # requirement so a specific label-scope question outranks the broader
        # Rule 15.1/15.2 goto family.
        if (
            re.search(r"\bgoto\b", clean)
            and re.search(r"\blabels?\b", clean)
            and re.search(r"\b(?:scope|block|same|enclos|restrict|where|location|declare|declared)\w*\b", clean)
        ):
            add("any label referenced by a goto statement shall be declared in the same block, or in any block enclosing the goto statement")

        # Text-only rationale questions can name the side-effect concept directly
        # without saying "function call".  Recognize the right/RHS/kanan operand
        # plus logical operator and preserve Rule 13.5 source wording.
        if (
            re.search(r"\b(?:side[- ]?effects?|persistent\s+side[- ]?effects?)\b", clean)
            and (
                re.search(r"\b(?:right[- ]hand\s+(?:side|operand)|right\s+side|rhs|kanan)\b", clean)
                or "&&" in raw
                or "||" in raw
            )
            and ("&&" in raw or "||" in raw or re.search(r"\blogical\s+(?:and|or)\b", clean))
        ):
            add("the right hand operand of a logical && or || operator shall not contain persistent side effects")

        # Pointers returned by Standard Library functions have two distinct
        # source requirements.  Preserve both when the user's topic is broad.
        if (
            re.search(r"\bpointers?\b", clean)
            and (
                re.search(r"\bstandard\s+library\b", clean)
                or re.search(r"\b(?:galing|mula)\s+(?:sa\s+)?standard\s+library\b", clean)
            )
        ):
            add("the pointers returned by the Standard Library functions localeconv getenv setlocale or strerror shall only be used as if they have pointer to const-qualified type")
            add("the pointer returned by the Standard Library functions asctime ctime gmtime localtime localeconv getenv setlocale or strerror shall not be used following a subsequent call to the same function")

        # Exact concept intersection for string handling + pointer bounds.
        if (
            re.search(r"\bstring(?:\s+handling)?\b|<string\.h>", clean)
            and re.search(r"\b(?:bound|bounds|beyond|out[- ]of[- ]bounds?)\b", clean)
            and re.search(r"\bpointers?\b|\bparameters?\b", clean)
        ):
            add("use of the string handling functions from string h shall not result in accesses beyond the bounds of the objects referenced by their pointer parameters")

        if re.search(r"(?:<\s*stdarg\.h\s*>|\bstdarg(?:\.h)?\b|\bva_(?:start|arg|end|copy)\b)", raw, re.IGNORECASE):
            add("the features of stdarg h shall not be used")

        # English-first natural switch questions. Keep these source-language
        # cues identifier-free so the authoritative corpus still decides which
        # Rule body is returned. These patterns cover ordinary reviewer wording
        # that does not paste literal C syntax. Concrete C snippets are handled
        # by the structure-aware code path below; keeping prose cues out of code
        # avoids broadening narrow historical code-review behavior merely because
        # a pasted snippet happens to contain ``default:`` or another token.
        if (
            not cls.looks_like_c_cpp(raw)
            and re.search(r"\bswitch(?:[- ]expression|\s+expression|\s+statement|\s+statements)?\b", clean)
        ):
            if re.search(r"\b(?:bool|boolean|essentially\s+boolean)\b", clean):
                add("a switch-expression shall not have essentially Boolean type")
            if re.search(r"\bdefault(?:\s+label|\s+case)?\b", clean):
                add("every switch statement shall have a default label")
                if re.search(r"\b(?:first|last|position|placement|where|order)\b", clean):
                    add("a default label shall appear as either the first or the last switch label of a switch statement")
            if (
                re.search(r"\b(?:one|1)\s+case\b", clean)
                and re.search(r"\b(?:one|1|a)\s+default\b|\bdefault(?:\s+label|\s+case)?\b", clean)
            ) or re.search(r"\b(?:two|2)\s+switch[- ]clauses?\b", clean):
                add("every switch statement shall have at least two switch-clauses")

        # "default label" is itself precise MISRA switch vocabulary.  Follow-up
        # turns often omit the word "switch" (for example "What if the default
        # label is first instead?").  Preserve source-language grounding without
        # depending on stale conversation text.
        if not cls.looks_like_c_cpp(raw) and re.search(r"\bdefault\s+label\b", clean):
            add("every switch statement shall have a default label")
            if re.search(r"\b(?:first|last|position|placement|where|order)\b", clean):
                add("a default label shall appear as either the first or the last switch label of a switch statement")

        # Natural empty terminating-else question.  The exact Rule body says
        # the if/else-if chain must terminate with else; its attached
        # Amplification states that the else must contain a side effect or a
        # comment.  Emit only the parent source statement here so structured
        # rescue remains identifier-free and source-grounded.
        if (
            re.search(r"\b(?:empty|blank)\b.{0,24}\belse\b", clean)
            or re.search(r"\belse\b.{0,24}\b(?:empty|blank)\b", clean)
            or re.search(r"\belse\s+block\b.{0,24}\b(?:nothing|no\s+action)\b", clean)
        ):
            add("all if else if constructs shall be terminated with an else statement")

        # Broad English control-flow questions should retrieve a small, useful
        # cross-section of authoritative control-flow requirements instead of
        # falling back merely because no exact Rule number was named.
        if re.search(r"\bcontrol[- ]flow\b", clean) and re.search(
            r"\b(?:misra|rule|rules|guideline|guidelines|mistake|mistakes|issue|issues|check|checks)\b",
            clean,
        ):
            add("the goto statement should not be used")
            add("the body of an iteration-statement or a selection-statement shall be a compound-statement")
            add("all if else if constructs shall be terminated with an else statement")
            add("every switch statement shall have a default label")

        # A text-only reviewer question can describe the logical-RHS case
        # without pasting a concrete call expression. Treat that description
        # as an independent semantic scenario instead of borrowing an older
        # conversation anchor. This emits source wording only; no Rule number
        # is injected here.
        if (
            re.search(
                r"\b(?:function\s+call|call\s+to\s+(?:a\s+)?function|"
                r"function\s+invocation|tawag\s+sa\s+function)\b",
                clean,
            )
            and re.search(r"\b(?:right[- ]hand\s+(?:side|operand)|right\s+side|rhs)\b", clean)
            and (
                "&&" in raw
                or "||" in raw
                or re.search(r"\blogical\s+(?:and|or)\b", clean)
            )
        ):
            add("the right hand operand of a logical && or || operator shall not contain persistent side effects")

        if re.search(r"\b(?:unreachable|after\s+return|code\s+after\s+return)\b", clean):
            add("a project shall not contain unreachable code")

        if re.search(r"\b(?:dead\s+code|dead\s+store|useless\s+code|no\s+effect|walang\s+effect)\b", clean):
            add("there shall be no dead code")

        if re.search(r"\b(?:signed|negative)\b.{0,35}\b(?:shift|shifting|sini[- ]?shift)\b", clean):
            add("operands shall not be of an inappropriate essential type")

        if (
            re.search(
                r"\b(?:recursion|recursive|calls?\s+(?:itself|themselves)|"
                r"self[- ]?call(?:ing)?|functions?\s+(?:from\s+)?(?:calls?|calling)\s+(?:itself|themselves)|"
                r"sarili(?:ng)?\s+function)\b",
                clean,
            )
            or re.search(
                r"\bfunction\b.{0,45}\b(?:sarili\s+niya|sarili)\b",
                clean,
            )
            or re.search(
                r"\b(?:sarili\s+niya|sarili)\b.{0,45}\bfunction\b",
                clean,
            )
        ):
            add("functions shall not call themselves either directly or indirectly")

        dynamic_memory = bool(
            re.search(r"\b(?:dynamic(?:\s+heap)?\s+memory|malloc|calloc|realloc|free)\b", clean)
        )
        if dynamic_memory:
            add("dynamic heap memory allocation shall not be used")
            if (
                re.search(r"\b(?:malloc|calloc|realloc|free)\b", clean)
                or re.search(
                    r"\b(?:applicable|all|every|lahat|mga)\b.{0,24}\b(?:requirements?|rules?|guidelines?)\b",
                    clean,
                )
                or re.search(r"\b(?:requirements?|rules?)\b.{0,24}\b(?:applicable|related)\b", clean)
            ):
                add(
                    "the memory allocation and deallocation functions of stdlib h shall not be used"
                )

        return output

    @classmethod
    def semantic_cues(cls, question: str) -> list[str]:
        """Describe visible C/C++ constructs without guessing Rule numbers.

        These cues are deliberately syntax/concept based rather than a mapping
        from a canned QA sentence to a known MISRA identifier.  They give BM25,
        vector search, and the reranker source-language terms that can match
        the actual rule statements while still allowing arbitrary natural
        wording and pasted code.
        """

        raw = str(question or "")
        code = cls._code_only_view(raw) or raw
        clean = re.sub(r"\s+", " ", raw).casefold()
        cues: list[str] = []

        def add(value: str) -> None:
            if value and value not in cues:
                cues.append(value)

        # First interpret explicit human concepts.  The syntax/data-flow layer
        # below can then add or refine the same source-language requirements.
        for natural_cue in cls._natural_requirement_cues(raw):
            add(natural_cue)

        # Assignment used as the value of a controlling expression.  Avoid
        # equality/comparison operators; the intent here is the single '='
        # assignment operator visible inside if/while conditions.
        if re.search(
            r"\b(?:if|while)\s*\([^)]*(?<![=!<>])=(?!=)[^)]*\)",
            raw,
            re.IGNORECASE | re.DOTALL,
        ):
            add("the result of an assignment operator should not be used")
            add("the controlling expression of an if statement shall have essentially Boolean type")

        if (
            re.search(r"\bgoto\b", clean)
            and not (
                re.search(r"\blabels?\b", clean)
                and re.search(
                    r"\b(?:scope|block|same|enclos|restrict|where|location|declare|declared)\w*\b",
                    clean,
                )
            )
        ):
            add("the goto statement should not be used")

        if re.search(
            r"(?:__attribute__|__declspec|__asm__|\basm\b|#\s*pragma|"
            r"compiler[- ]specific|language extension|compiler extension)",
            raw,
            re.IGNORECASE,
        ):
            add("language extensions should not be used")

        # Signed/unsigned initialization or assignment where a visibly negative
        # expression is stored in an unsigned object.  Use the vocabulary of
        # the essential type rules without deciding the Rule number here.
        if (
            re.search(
                r"\bunsigned\b[^;\n=]*\b[A-Za-z_]\w*\s*=\s*-\s*(?:\d+|[A-Za-z_]\w*)",
                raw,
                re.IGNORECASE,
            )
            or ("unsigned" in clean and re.search(r"\bnegative\b", clean))
        ):
            add(
                "the value of an expression shall not be assigned to an object "
                "with a different essential type category"
            )
            add("non-negative integer constant expression unsigned type representable")

        if "<<" in raw or ">>" in raw:
            add("shift operator operands essentially unsigned type shift range")

        if re.search(r"(?<![&])&(?![&])|(?<![|])\|(?![|])|\^|~", raw):
            add("bitwise operator operands essential type unsigned")

        # Rule 13.5 applies only when the right-hand operand of &&/|| has a
        # persistent side effect. Merely containing a logical operator is not a
        # violation. Keep this deliberately conservative so expressions such as
        # ``(x++ > 2) && (x < 10)`` are not falsely flagged.
        for logical in re.finditer(r"(?:&&|\|\|)(?P<rhs>[^;\n]+)", raw):
            rhs = logical.group("rhs")
            rhs_has_assignment = bool(re.search(r"(?<![=!<>])=(?!=)", rhs))
            rhs_has_incdec = "++" in rhs or "--" in rhs
            rhs_has_call = bool(re.search(
                r"\b(?!(?:if|for|while|switch|sizeof)\b)[A-Za-z_]\w*\s*\(",
                rhs,
                re.IGNORECASE,
            ))
            if rhs_has_assignment or rhs_has_incdec or rhs_has_call:
                add("the right hand operand of a logical && or || operator shall not contain persistent side effects")
                break

        # Increment/decrement alone is not a violation.  Only enrich when the
        # same semicolon-delimited full expression visibly contains another
        # likely side effect (assignment or function call), avoiding false
        # candidate pressure from harmless ``x++;`` statements.
        for statement in re.split(r";|\n", raw):
            if "++" not in statement and "--" not in statement:
                continue
            # A semicolon-delimited slice can still include enclosing syntax
            # such as ``switch (x) { case 1: x++``.  Only inspect the visible
            # full-expression fragment after the last block/label delimiter
            # so the controlling ``switch(...)`` is not mistaken for another
            # side effect of ``x++``.
            expression_fragment = re.split(r"[{}:]", statement)[-1]
            remainder = re.sub(r"\+\+|--", "", expression_fragment)
            has_assignment = bool(
                re.search(r"(?<![=!<>])=(?!=)", remainder)
            )
            has_call = bool(
                re.search(
                    r"\b(?!(?:if|for|while|switch|sizeof|return)\b)[A-Za-z_]\w*\s*\(",
                    remainder,
                    re.IGNORECASE,
                )
            )
            if has_assignment or has_call:
                add(
                    "a full expression containing an increment or decrement "
                    "operator should have no other potential side effects"
                )
                break

        # Increment/decrement inside a controlling expression is a natural
        # reason to retrieve Rule 13.3 even when the construct is compliant.
        # Final assessment below distinguishes whether another side effect is
        # present in the same full expression.
        if re.search(r"\b(?:if|while)\s*\([^\n]*(?:\+\+|--)[^\n]*\)", code, re.IGNORECASE):
            add(
                "a full expression containing an increment or decrement operator should have no other potential side effects other than that caused by the increment or decrement operator"
            )

        # v6.4.36 manual-generalization recovery.  These detectors are
        # syntax-driven and emit the exact wording of source requirements;
        # they do not map user questions to hard-coded Rule numbers.

        # Multiple visible return statements in one function body indicate
        # more than one exit point.  Extract balanced function bodies so an
        # early return nested inside an if-block is still visible.
        for function_definition in re.finditer(
            r"\b(?:[A-Za-z_]\w*\s+|[A-Za-z_]\w*\s*\*\s*)+"
            r"(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{",
            raw,
            re.IGNORECASE,
        ):
            brace_start = function_definition.end() - 1
            depth = 0
            body_end = len(raw)
            for index in range(brace_start, len(raw)):
                character = raw[index]
                if character == "{":
                    depth += 1
                elif character == "}":
                    depth -= 1
                    if depth == 0:
                        body_end = index
                        break
            function_body = raw[brace_start + 1:body_end]
            return_focus = bool(
                re.search(
                    r"(?:pag[- ]?return|return\s+statements?|exit\s+points?|single\s+point\s+of\s+exit)",
                    raw,
                    re.IGNORECASE,
                )
            )
            if (
                return_focus
                and len(re.findall(r"\breturn\b", function_body, re.IGNORECASE)) > 1
            ):
                add("a function should have a single point of exit at the end")
                break

        # Inspect each visible switch body for two direct structural issues:
        # a missing default label and a switch-clause that visibly falls
        # through without an unconditional break.  This is deliberately
        # conservative; ambiguous nested-control-flow cases remain on the
        # configured complex-model path.
        for switch_match in re.finditer(r"\bswitch\s*\([^)]*\)\s*\{", raw, re.IGNORECASE):
            brace_start = switch_match.end() - 1
            depth = 0
            body_end = len(raw)
            for index in range(brace_start, len(raw)):
                character = raw[index]
                if character == "{":
                    depth += 1
                elif character == "}":
                    depth -= 1
                    if depth == 0:
                        body_end = index
                        break
            switch_body = raw[brace_start + 1:body_end]

            # Include satisfied companion requirements for explicit broad
            # "relevant checks/rules" reviews, while preserving the narrower
            # historical violation-review behavior. This avoids flooding a
            # simple "is there a violation?" question with every satisfied
            # switch rule, but gives broader code-review prompts a useful
            # multi-rule picture.
            broad_switch_review = bool(re.search(
                r"\b(?:relevant|applicable|all|every)\b.{0,35}\b(?:checks?|rules?|requirements?)\b"
                r"|\b(?:which|what)\b.{0,35}\b(?:checks?|rules?|requirements?)\b",
                raw,
                re.IGNORECASE,
            ))
            has_default = bool(re.search(r"\bdefault\s*:", switch_body, re.IGNORECASE))
            if not has_default or broad_switch_review:
                add("every switch statement shall have a default label")

            labels = list(re.finditer(r"\b(?:case\b[^:]*|default)\s*:", switch_body, re.IGNORECASE))
            # Preserve the historical narrow-review contract: report Rule 16.6
            # when it is actually violated, or when the user explicitly asks
            # for a broad set of relevant/applicable checks. A narrow review
            # with two-or-more clauses should not be padded with an unrelated
            # satisfied companion rule.
            if labels and (len(labels) < 2 or broad_switch_review):
                add("every switch statement shall have at least two switch-clauses")
            if has_default and labels:
                default_positions = [
                    index for index, label in enumerate(labels)
                    if re.match(r"\s*default\b", label.group(0), re.IGNORECASE)
                ]
                misplaced_default = bool(
                    default_positions
                    and default_positions[0] not in {0, len(labels) - 1}
                )
                if broad_switch_review or misplaced_default:
                    add("a default label shall appear as either the first or the last switch label of a switch statement")

            for label_index, label in enumerate(labels):
                clause_start = label.end()
                clause_end = labels[label_index + 1].start() if label_index + 1 < len(labels) else len(switch_body)
                clause = switch_body[clause_start:clause_end]
                # Empty grouped labels are allowed to fall through; require a
                # break only when the clause visibly contains executable text.
                stripped_clause = re.sub(r"/\*.*?\*/|//[^\n]*", "", clause, flags=re.DOTALL).strip()
                if not stripped_clause:
                    continue
                if not re.search(r"\bbreak\s*;", stripped_clause, re.IGNORECASE):
                    add("an unconditional break statement shall terminate every switch-clause")
                    break

        # String literals assigned to a non-const char pointer are directly
        # visible and can be assessed without inferring hidden declarations.
        if re.search(
            r"(?m)^\s*(?!const\b)(?:static\s+|extern\s+|volatile\s+)*"
            r"char\s*\*\s*[A-Za-z_]\w*\s*=\s*\"",
            raw,
            re.IGNORECASE,
        ):
            add(
                "a string literal shall not be assigned to an object unless the object's "
                "type is pointer to const-qualified char"
            )

        # Old-style empty parameter lists such as ``extern int f();`` are not
        # prototype-form declarations.  Function calls do not match because a
        # declaration type must be visible before the function name.
        if re.search(
            r"(?m)^\s*(?:extern\s+|static\s+)?"
            r"(?:const\s+|volatile\s+)*(?:signed\s+|unsigned\s+)?"
            r"(?:void|char|short|int|long|float|double|u?int\d+_t|[A-Za-z_]\w*_t)"
            r"(?:\s+long)?\s+\**\s*[A-Za-z_]\w*\s*\(\s*\)\s*;",
            raw,
            re.IGNORECASE,
        ):
            add("function types shall be in prototype form with named parameters")

        # A bitwise operation whose visible operands are declared as signed
        # integer types grounds the essential-type restriction.  Unsigned-only
        # bitwise expressions are not flagged by this detector.
        signed_type_pattern = (
            r"(?:signed\s+)?(?:char|short|int|long|int\d+_t|int_least\d+_t|int_fast\d+_t)"
        )
        for bitwise in re.finditer(
            r"\b(?P<left>[A-Za-z_]\w*)\s*(?:&|\||\^)\s*(?P<right>[A-Za-z_]\w*)\b",
            raw,
        ):
            before_operator = raw[:bitwise.start()]
            signed_visible = False
            for operand_name in (bitwise.group("left"), bitwise.group("right")):
                declaration = re.search(
                    rf"\b(?P<type>{signed_type_pattern})\s+{re.escape(operand_name)}\b",
                    before_operator,
                    re.IGNORECASE,
                )
                unsigned_declaration = re.search(
                    rf"\bunsigned\s+(?:char|short|int|long)\s+{re.escape(operand_name)}\b|"
                    rf"\buint\d+_t\s+{re.escape(operand_name)}\b",
                    before_operator,
                    re.IGNORECASE,
                )
                if declaration and not unsigned_declaration:
                    signed_visible = True
                    break
            if signed_visible:
                add("operands shall not be of an inappropriate essential type")
                break

        # Visible single-parameter call compatibility.  When a prototype and
        # argument declaration are both present, compare only the signedness
        # category that is explicit in the snippet; hidden types remain
        # ambiguous and therefore use the normal LLM/retrieval path.
        integer_type = (
            r"(?:(?:const|volatile)\s+)*(?:(?:signed|unsigned)\s+)?"
            r"(?:char|short|int|long|u?int\d+_t|size_t|intptr_t|uintptr_t)"
        )

        def visible_integer_category(type_text: str) -> str:
            value = re.sub(r"\s+", " ", str(type_text or "").strip()).casefold()
            if "unsigned" in value or re.search(r"\buint\d+_t\b", value):
                return "unsigned"
            if re.search(r"\b(?:int\d+_t|intptr_t|char|short|int|long)\b", value):
                return "signed"
            return ""

        for prototype in re.finditer(
            rf"(?m)^\s*(?:extern\s+|static\s+)?(?:[A-Za-z_]\w*\s+)+"
            rf"(?P<function>[A-Za-z_]\w*)\s*\(\s*(?P<ptype>{integer_type})\s+"
            rf"[A-Za-z_]\w*\s*\)\s*;",
            raw,
            re.IGNORECASE,
        ):
            function_name = prototype.group("function")
            parameter_category = visible_integer_category(prototype.group("ptype"))
            if not parameter_category:
                continue
            for call in re.finditer(
                rf"\b{re.escape(function_name)}\s*\(\s*(?P<arg>[A-Za-z_]\w*)\s*\)\s*;",
                raw[prototype.end():],
                re.IGNORECASE,
            ):
                absolute_call_start = prototype.end() + call.start()
                argument_name = call.group("arg")
                before_call = raw[:absolute_call_start]
                argument_declarations = list(re.finditer(
                    rf"\b(?P<atype>{integer_type})\s+{re.escape(argument_name)}\b",
                    before_call,
                    re.IGNORECASE,
                ))
                if not argument_declarations:
                    continue
                argument_category = visible_integer_category(
                    argument_declarations[-1].group("atype")
                )
                if argument_category and argument_category != parameter_category:
                    add(
                        "the value of an expression shall not be assigned to an object with a "
                        "narrower essential type or of a different essential type category"
                    )
                    break

        # Integer zero used directly as a pointer initializer is a visible
        # integer null pointer constant; NULL remains intentionally unflagged.
        if re.search(
            r"(?m)^\s*(?:const\s+|volatile\s+|static\s+|extern\s+)*"
            r"(?:struct\s+[A-Za-z_]\w*|union\s+[A-Za-z_]\w*|void|char|short|int|long|"
            r"float|double|u?int\d+_t|[A-Za-z_]\w*_t)\s*\*+\s*"
            r"[A-Za-z_]\w*\s*=\s*0(?:[uUlL]*)\s*;",
            raw,
            re.IGNORECASE,
        ):
            add("the macro NULL shall be the only permitted form of integer null pointer constant")

        # An ``if ... else if`` chain with no final ``else`` is directly
        # decidable from the pasted construct.
        if re.search(r"\belse\s+if\b", raw, re.IGNORECASE):
            has_terminal_else = bool(
                re.search(r"\belse\b(?!\s*if\b)", raw, re.IGNORECASE)
            )
            if not has_terminal_else:
                add("all if else if constructs shall be terminated with an else statement")

        if re.search(r"\bswitch\s*\(", raw, re.IGNORECASE):
            add("switch statement controlling expression case label")

        if re.search(
            r"\b(?:malloc|calloc|realloc|free)\s*\(",
            raw,
            re.IGNORECASE,
        ):
            # Two independently citable MISRA requirements cover this visible
            # construct: the directive banning dynamic allocation and the rule
            # banning the Standard Library allocation/deallocation functions.
            # Use source-language requirement text rather than identifiers so
            # this remains corpus-driven rather than test-question-driven.
            add("dynamic memory allocation shall not be used")
            add(
                "the memory allocation and deallocation functions of stdlib h "
                "shall not be used"
            )

        # Detect direct self-recursion from a visible function definition.
        # Use balanced braces instead of a non-greedy regex so nested if/for
        # blocks cannot hide a self-call that appears later in the function.
        for function_definition in re.finditer(
            r"\b(?:[A-Za-z_]\w*\s+|[A-Za-z_]\w*\s*\*\s*)+"
            r"(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{",
            raw,
            re.IGNORECASE,
        ):
            function_name = function_definition.group("name")
            brace_start = function_definition.end() - 1
            depth = 0
            body_end = len(raw)
            for index in range(brace_start, len(raw)):
                character = raw[index]
                if character == "{":
                    depth += 1
                elif character == "}":
                    depth -= 1
                    if depth == 0:
                        body_end = index
                        break
            function_body = raw[brace_start + 1:body_end]
            if re.search(
                rf"\b{re.escape(function_name)}\s*\(",
                function_body,
                re.IGNORECASE,
            ):
                add("functions shall not call themselves either directly or indirectly")
                break

        if re.search(r"(?m)^\s*#\s*undef\b", raw, re.IGNORECASE):
            add("undef should not be used")

        c_keywords = (
            "auto|break|case|char|const|continue|default|do|double|else|enum|"
            "extern|float|for|goto|if|inline|int|long|register|restrict|return|"
            "short|signed|sizeof|static|struct|switch|typedef|union|unsigned|"
            "void|volatile|while|_bool|_complex|_imaginary"
        )
        if re.search(
            rf"(?m)^\s*#\s*define\s+(?:{c_keywords})\b",
            raw,
            re.IGNORECASE,
        ):
            add("a macro shall not be defined with the same name as a keyword")

        numeric_if = re.search(
            r"(?m)^\s*#\s*(?:if|elif)\s+([+-]?\d+)\b",
            raw,
            re.IGNORECASE,
        )
        numeric_if_value = None
        if numeric_if:
            try:
                numeric_if_value = int(numeric_if.group(1), 10)
            except ValueError:
                numeric_if_value = None

        # Also resolve the common visible pattern ``#define NAME <integer>``
        # followed by ``#if NAME``.  This stays source-syntax-driven and does
        # not map any canned question to a Rule number.
        if numeric_if_value is None:
            integer_macros = {
                match.group("name"): int(match.group("value"), 10)
                for match in re.finditer(
                    r"(?m)^\s*#\s*define\s+(?P<name>[A-Za-z_]\w*)\s+"
                    r"(?P<value>[+-]?\d+)\b",
                    raw,
                    re.IGNORECASE,
                )
            }
            symbolic_if = re.search(
                r"(?m)^\s*#\s*(?:if|elif)\s+(?P<name>[A-Za-z_]\w*)\b",
                raw,
                re.IGNORECASE,
            )
            if symbolic_if:
                numeric_if_value = integer_macros.get(symbolic_if.group("name"))

        if numeric_if_value is not None and numeric_if_value not in {0, 1}:
            add(
                "the controlling expression of a #if or #elif preprocessing "
                "directive shall evaluate to 0 or 1"
            )

        if re.search(r"\bunion\b", raw, re.IGNORECASE):
            add("the union keyword should not be used")

        # Object-pointer/integer conversion.  Support both a literal operand
        # and an identifier whose visible declaration is an integer type.
        # This is intentionally syntax-driven and does not infer hidden types.
        pointer_cast = re.search(
            r"\(\s*(?:const\s+|volatile\s+|signed\s+|unsigned\s+)*"
            r"(?:char|short|int|long|float|double|struct\s+[A-Za-z_]\w*|"
            r"union\s+[A-Za-z_]\w*|[A-Za-z_]\w*)\s*\*+\s*\)"
            r"\s*(?P<operand>0x[0-9a-f]+|\d+|[A-Za-z_]\w*)\b",
            raw,
            re.IGNORECASE,
        )
        if pointer_cast:
            operand = pointer_cast.group("operand")
            operand_is_integer = bool(re.fullmatch(r"(?:0x[0-9a-f]+|\d+)", operand, re.IGNORECASE))
            if not operand_is_integer:
                before_cast = raw[:pointer_cast.start()]
                integer_decl = re.search(
                    rf"\b(?:signed\s+|unsigned\s+)?"
                    rf"(?:char|short|int|long|size_t|u?int\d+_t|uintptr_t|intptr_t)"
                    rf"(?:\s+long)?\s+{re.escape(operand)}\b",
                    before_cast,
                    re.IGNORECASE,
                )
                operand_is_integer = bool(integer_decl)
            if operand_is_integer:
                add(
                    "a conversion should not be performed between a pointer to object "
                    "and an integer type"
                )

        # --------------------------------------------------------------
        # v6.4.37 generic C-construct interpretation layer
        # --------------------------------------------------------------
        # These checks operate on the complete multiline snippet. They derive
        # source-language MISRA concepts from syntax/data-flow that is visible
        # in the user's code. They are intentionally independent of any canned
        # natural-language question and do not insert Rule/Directive numbers.

        # Selection/iteration bodies must be compound statements. This catches
        # unbraced single statements while leaving already-braced bodies alone.
        if re.search(
            r"\b(?:if|for|while)\s*\([^\n]*?\)\s*(?!\{)(?:[A-Za-z_]|\*|\+\+|--)",
            code,
            re.IGNORECASE,
        ) or re.search(r"\belse\s*(?!if\b|\{)(?:[A-Za-z_]|\*|\+\+|--)", code, re.IGNORECASE):
            add("the body of an iteration-statement or a selection-statement shall be a compound-statement")

        # Rule 9.1 concept: distinguish automatic locals from file-scope/static
        # objects. A local declaration without an initializer followed only by
        # conditional assignment before a later read is a visible unsafe path.
        function_matches = list(re.finditer(
            r"\b(?:[A-Za-z_]\w*\s+|[A-Za-z_]\w*\s*\*\s*)+"
            r"(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{",
            code,
            re.IGNORECASE,
        ))
        function_ranges = []
        for fm in function_matches:
            start = fm.end() - 1
            depth = 0
            end = len(code)
            for idx in range(start, len(code)):
                if code[idx] == "{":
                    depth += 1
                elif code[idx] == "}":
                    depth -= 1
                    if depth == 0:
                        end = idx
                        break
            function_ranges.append((fm.start(), start + 1, end, code[start + 1:end]))

        automatic_unset_visible = False
        scalar_decl = (
            r"(?m)^\s*(?P<type>(?:(?:const|volatile|signed|unsigned)\s+)*"
            r"(?:char|short|int|long|float|double|size_t|u?int\d+_t|intptr_t|uintptr_t))"
            r"\s+(?P<name>[A-Za-z_]\w*)\s*;"
        )
        for _func_start, body_start, _body_end, body in function_ranges:
            for decl in re.finditer(scalar_decl, body, re.IGNORECASE):
                name = decl.group("name")
                tail = body[decl.end():]
                # Direct read-like uses: function argument, arithmetic/comparison,
                # return, array index, or RHS occurrence. Pure LHS assignment is
                # ignored.
                occurrences = list(re.finditer(
                    rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", tail
                ))
                if not occurrences:
                    continue
                read_match = None
                for occurrence in occurrences:
                    occurrence_tail = tail[occurrence.start():]
                    # Ignore declaration-like/LHS assignment occurrences.
                    if re.match(rf"{re.escape(name)}\s*=(?!=)", occurrence_tail):
                        continue
                    read_match = occurrence
                    break
                if read_match is None:
                    continue
                before_use = tail[:read_match.start()]
                # An assignment is guaranteed only when it occurs at the local
                # function-body depth before the read. Assignments inside an if/
                # loop body do not prove all paths set the object.
                depth = 0
                unconditional_set = False
                for line in before_use.splitlines():
                    if depth == 0 and re.search(
                        rf"\b{re.escape(name)}\s*=(?!=)", line
                    ):
                        unconditional_set = True
                        break
                    depth += line.count("{") - line.count("}")
                if not unconditional_set:
                    automatic_unset_visible = True
                    break
            if automatic_unset_visible:
                break
        if automatic_unset_visible:
            add("the value of an object with automatic storage duration shall not be read before it has been set")

        # A file-scope scalar without an explicit initializer is still useful to
        # answer a natural 'used before set' question. Rule 9.1's own rationale
        # states that static-storage objects are automatically initialized to zero.
        first_function_start = function_ranges[0][0] if function_ranges else len(code)
        file_scope_prefix = code[:first_function_start]
        if re.search(scalar_decl, file_scope_prefix, re.IGNORECASE) and re.search(
            r"(?:before\s+(?:it\s+)?(?:is\s+)?set|before\s+set|bago\s+ma[- ]?set|bago\s+set)",
            clean,
            re.IGNORECASE,
        ):
            add("the value of an object with automatic storage duration shall not be read before it has been set")

        # Known-size array indexing with a visibly constant out-of-range index.
        array_decl_matches = list(re.finditer(
            r"\b(?:char|short|int|long|float|double|u?int\d+_t|[A-Za-z_]\w*_t)"
            r"\s+(?P<name>[A-Za-z_]\w*)\s*\[\s*(?P<size>\d+)\s*\]",
            code,
            re.IGNORECASE,
        ))
        array_sizes = {
            m.group("name"): int(m.group("size"))
            for m in array_decl_matches
        }
        array_decl_spans = [(m.start(), m.end()) for m in array_decl_matches]
        integer_values = {
            m.group("name"): int(m.group("value"))
            for m in re.finditer(
                r"\b(?:signed\s+|unsigned\s+)?(?:char|short|int|long|u?int\d+_t|size_t)"
                r"\s+(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>-?\d+)[uUlL]*\s*;",
                raw,
                re.IGNORECASE,
            )
        }
        out_of_range_index = False
        for access in re.finditer(r"\b(?P<array>[A-Za-z_]\w*)\s*\[\s*(?P<index>-?\d+|[A-Za-z_]\w*)\s*\]", code):
            # Do not mistake an array declaration such as ``int data[10]`` for
            # an access of element 10.  This false positive caused Rule 18.1 to
            # be marked non-compliant for otherwise in-range pointer arithmetic.
            if any(start <= access.start() < end for start, end in array_decl_spans):
                continue
            array = access.group("array")
            if array not in array_sizes:
                continue
            token = access.group("index")
            try:
                index_value = int(token)
            except ValueError:
                index_value = integer_values.get(token)
            if index_value is not None and not (0 <= index_value < array_sizes[array]):
                out_of_range_index = True
                break
        if out_of_range_index:
            add("a pointer resulting from arithmetic on a pointer operand shall address an element of the same array as that pointer operand")

        # Pointer arithmetic operators +, -, += and -= on visibly declared
        # pointer objects. Increment/decrement are intentionally excluded because
        # Rule 18.4 does not prohibit them.
        pointer_names = {
            m.group("name")
            for m in re.finditer(
                r"\b(?:const\s+|volatile\s+|static\s+|extern\s+)*"
                r"(?:void|char|short|int|long|float|double|u?int\d+_t|struct\s+\w+|union\s+\w+|[A-Za-z_]\w*_t)"
                r"\s*\*+\s*(?P<name>[A-Za-z_]\w*)",
                raw,
                re.IGNORECASE,
            )
        }
        pointer_arithmetic = any(
            re.search(rf"\b{re.escape(name)}\s*(?:\+|-|\+=|-=)\s*", code)
            or re.search(rf"(?:\+|-)\s*\b{re.escape(name)}\b", code)
            for name in pointer_names
        )
        if pointer_arithmetic:
            add("the +, -, += and -= operators should not be applied to an expression of pointer type")

        # If a visible pointer is based on a known local array and the constant
        # offset remains within that same array (or one-past for pointer
        # creation), Rule 18.1 is relevant but satisfied.  This lets the final
        # answer distinguish "applicable and satisfied" from Rule 18.4's
        # separate advisory restriction on explicit pointer arithmetic.
        pointer_array_bases: dict[str, tuple[str, int]] = {}
        for base in re.finditer(
            r"\b(?:const\s+|volatile\s+|static\s+|extern\s+)*"
            r"(?:void|char|short|int|long|float|double|u?int\d+_t|[A-Za-z_]\w*_t)"
            r"\s*\*+\s*(?P<pointer>[A-Za-z_]\w*)\s*=\s*(?P<array>[A-Za-z_]\w*)\s*;",
            code,
            re.IGNORECASE,
        ):
            array_name = base.group("array")
            if array_name in array_sizes:
                pointer_array_bases[base.group("pointer")] = (array_name, array_sizes[array_name])

        for pointer_name, (_array_name, array_size) in pointer_array_bases.items():
            arithmetic = re.search(
                rf"\b{re.escape(pointer_name)}\s*=\s*{re.escape(pointer_name)}\s*(?P<op>[+-])\s*(?P<offset>\d+)\s*;",
                code,
                re.IGNORECASE,
            )
            if not arithmetic:
                continue
            offset = int(arithmetic.group("offset"))
            if arithmetic.group("op") == "-":
                offset = -offset
            if 0 <= offset <= array_size:
                add("a pointer resulting from arithmetic on a pointer operand shall address an element of the same array as that pointer operand")
                break

        # Signed left operand to a shift is an inappropriate essential type.
        signed_names = {
            m.group("name")
            for m in re.finditer(
                r"\b(?:signed\s+)?(?:char|short|int|long|int\d+_t|intptr_t)\s+"
                r"(?P<name>[A-Za-z_]\w*)\b",
                raw,
                re.IGNORECASE,
            )
        }
        if any(re.search(rf"\b{re.escape(name)}\s*(?:<<|>>)", code) for name in signed_names):
            add("operands shall not be of an inappropriate essential type")

        # Unreachable statements after an unconditional return in the same block.
        if re.search(
            r"\breturn(?:\s+[^;]+)?\s*;\s*(?:/\*.*?\*/\s*|//[^\n]*\n\s*)*"
            r"(?!(?:case\b|default\b|}))[A-Za-z_]\w*\s*\(",
            code,
            re.IGNORECASE | re.DOTALL,
        ):
            add("a project shall not contain unreachable code")

        # Obvious dead stores: a local variable is assigned, overwritten before
        # any read, and never contributes to the returned expression afterwards.
        for fm_start, _body_start, _body_end, body in function_ranges:
            for first in re.finditer(r"\b(?P<name>[A-Za-z_]\w*)\s*=\s*[^;]+;", body):
                name = first.group("name")
                tail = body[first.end():]
                second = re.search(rf"\b{re.escape(name)}\s*=\s*[^;]+;", tail)
                if not second:
                    continue
                between = tail[:second.start()]
                if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", between):
                    continue
                after = tail[second.end():]
                if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", after):
                    add("there shall be no dead code")
                    break

        # for(;;) is explicitly allowed by the Rule 14.2 exception. A single
        # break used for early termination is also allowed by Rule 15.4. Emit
        # both source requirements so the answer can explain why this construct
        # is not inherently non-compliant.
        for infinite in re.finditer(r"\bfor\s*\(\s*;\s*;\s*\)\s*\{", code, re.IGNORECASE):
            start = infinite.end() - 1
            depth = 0
            end = len(code)
            for idx in range(start, len(code)):
                if code[idx] == "{":
                    depth += 1
                elif code[idx] == "}":
                    depth -= 1
                    if depth == 0:
                        end = idx
                        break
            loop_body = code[start + 1:end]
            add("a for loop shall be well-formed")
            if len(re.findall(r"\bbreak\s*;", loop_body, re.IGNORECASE)) <= 1:
                add("there should be no more than one break or goto statement used to terminate any iteration statement")
            break

        # Rule 16.6 code evidence is useful when it is violated or when the
        # user explicitly asks for a broad set of switch checks. Natural prose
        # such as "one case and one default" is handled by the English-first
        # semantic cue layer, so a narrow historical code review is not padded
        # with a satisfied companion requirement.
        broad_switch_review = bool(re.search(
            r"\b(?:relevant|applicable|all|every)\b.{0,35}\b(?:checks?|rules?|requirements?)\b"
            r"|\b(?:which|what)\b.{0,35}\b(?:checks?|rules?|requirements?)\b",
            question or "",
            re.IGNORECASE,
        ))
        for switch_match in re.finditer(r"\bswitch\s*\([^)]*\)\s*\{", code, re.IGNORECASE):
            start = switch_match.end() - 1
            depth = 0
            end = len(code)
            for idx in range(start, len(code)):
                if code[idx] == "{":
                    depth += 1
                elif code[idx] == "}":
                    depth -= 1
                    if depth == 0:
                        end = idx
                        break
            body = code[start + 1:end]
            label_matches = list(re.finditer(r"\b(?:case\b[^:]*|default)\s*:", body, re.IGNORECASE))
            labels = [match.group(0) for match in label_matches]
            has_default = bool(re.search(r"\bdefault\s*:", body, re.IGNORECASE))
            all_nonempty_clauses_terminated = True
            for label_index, label_match in enumerate(label_matches):
                clause_start = label_match.end()
                clause_end = label_matches[label_index + 1].start() if label_index + 1 < len(label_matches) else len(body)
                clause = re.sub(r"/\*.*?\*/|//[^\n]*", "", body[clause_start:clause_end], flags=re.DOTALL).strip()
                if clause and not re.search(r"\bbreak\s*;", clause, re.IGNORECASE):
                    all_nonempty_clauses_terminated = False
                    break
            if labels and (
                len(labels) < 2
                or broad_switch_review
                or (has_default and all_nonempty_clauses_terminated)
            ):
                add("every switch statement shall have at least two switch-clauses")

        # Pointer used directly as an if/while controlling expression is not
        # essentially Boolean. This is separate from integer scalar handling.
        for control in re.finditer(r"\b(?:if|while)\s*\(\s*(?P<name>[A-Za-z_]\w*)\s*\)", code, re.IGNORECASE):
            name = control.group("name")
            if name in pointer_names:
                add("the controlling expression of an if statement and the controlling expression of an iteration-statement shall have essentially Boolean type")
                break

        # Signed/unsigned operands in comparisons undergo the usual arithmetic
        # conversions; visibly different essential categories ground Rule 10.4.
        unsigned_names = {
            m.group("name")
            for m in re.finditer(
                r"\b(?:unsigned\s+(?:char|short|int|long)|uint\d+_t|uintptr_t|size_t)\s+"
                r"(?P<name>[A-Za-z_]\w*)\b", code, re.IGNORECASE
            )
        }
        for cmp_match in re.finditer(r"\b(?P<a>[A-Za-z_]\w*)\s*(?:<|>|<=|>=|==|!=)\s*(?P<b>[A-Za-z_]\w*)\b", code):
            a, b = cmp_match.group("a"), cmp_match.group("b")
            if (a in signed_names and b in unsigned_names) or (b in signed_names and a in unsigned_names):
                add("both operands of an operator in which the usual arithmetic conversions are performed shall have the same essential type category")
                break

        # Function-like macro definitions are a strong Rule 20.7 semantic cue
        # even when the current turn shows only the *corrected/protected*
        # definition and no invocation.  Retrieval must therefore happen for
        # both unsafe and visibly protected parameter-expression occurrences;
        # the deterministic assessment below decides the polarity.
        for definition in cls._function_macro_definitions(raw):
            if definition.get("expression_params"):
                add("expressions resulting from the expansion of macro parameters shall be enclosed in parentheses")
                break

        # Non-void function with an explicit return on only one branch and no
        # terminal return after the conditional has a visible exit path without
        # a return expression. Keep this conservative to simple, decidable code.
        for fm in function_matches:
            header = code[fm.start():fm.end()]
            if re.search(r"\bvoid\s+" + re.escape(fm.group("name")) + r"\s*\(", header, re.IGNORECASE):
                continue
            start = fm.end() - 1
            depth = 0
            end = len(code)
            for idx in range(start, len(code)):
                if code[idx] == "{":
                    depth += 1
                elif code[idx] == "}":
                    depth -= 1
                    if depth == 0:
                        end = idx
                        break
            body = code[start + 1:end].strip()
            if re.search(r"\breturn\s+[^;]+;", body) and re.search(r"\bif\s*\(", body):
                tail_after_last_return = re.split(r"\breturn\s+[^;]+;", body, flags=re.IGNORECASE)[-1]
                # If the last return is nested and the function body closes
                # without another return, a fall-through exit remains visible.
                if tail_after_last_return.count("}") >= 1 and not re.search(r"\breturn\s+[^;]+;", tail_after_last_return):
                    add("all exit paths from a function with non- void return type shall have an explicit return statement with an expression")
                    break

        # A visibly typed scalar object or visible function result used as an
        # if/while controlling expression is enough to ground the
        # essential-Boolean requirement.  Keep the interpretation source-
        # syntax-driven: only declarations present in the user's snippet are
        # used, and no Rule number is injected here.
        for control in re.finditer(
            r"\b(?:if|while)\s*\(\s*(?P<neg>!\s*)?(?P<expr>[A-Za-z_]\w*(?:\s*\(\s*\))?)\s*\)",
            code,
            re.IGNORECASE,
        ):
            expr = re.sub(r"\s+", "", control.group("expr"))
            name = expr[:-2] if expr.endswith("()") else expr
            before_control = code[:control.start()]

            scalar_declaration = re.search(
                rf"\b(?P<type>(?:(?:signed|unsigned|const|volatile)\s+)*"
                rf"(?:char|short|int|long|float|double|size_t|u?int\d+_t|intptr_t|uintptr_t))"
                rf"\s+{re.escape(name)}\b",
                before_control,
                re.IGNORECASE,
            )
            pointer_declaration = re.search(
                rf"\b(?:const\s+|volatile\s+|static\s+|extern\s+)*"
                rf"(?:void|char|short|int|long|float|double|u?int\d+_t|[A-Za-z_]\w*_t)"
                rf"\s*\*+\s*{re.escape(name)}\b",
                before_control,
                re.IGNORECASE,
            )
            boolean_declaration = re.search(
                rf"\b(?:bool|_Bool|bool_t)\s+{re.escape(name)}\b",
                before_control,
                re.IGNORECASE,
            )

            bool_function = re.search(
                rf"\b(?:bool|_Bool|bool_t)\s+{re.escape(name)}\s*\([^;{{}}]*\)\s*;",
                before_control,
                re.IGNORECASE,
            )
            scalar_function = re.search(
                rf"\b(?:(?:signed|unsigned|const|volatile)\s+)*"
                rf"(?:char|short|int|long|float|double|size_t|u?int\d+_t|intptr_t|uintptr_t)"
                rf"\s+{re.escape(name)}\s*\([^;{{}}]*\)\s*;",
                before_control,
                re.IGNORECASE,
            )
            pointer_function = re.search(
                rf"\b(?:const\s+|volatile\s+|static\s+|extern\s+)*"
                rf"(?:void|char|short|int|long|float|double|u?int\d+_t|[A-Za-z_]\w*_t)"
                rf"\s*\*+\s*{re.escape(name)}\s*\([^;{{}}]*\)\s*;",
                before_control,
                re.IGNORECASE,
            )

            visible_boolean = bool(boolean_declaration or (expr.endswith("()") and bool_function))
            visible_nonboolean = bool(
                (expr.endswith("()") and (scalar_function or pointer_function))
                or (not expr.endswith("()") and (scalar_declaration or pointer_declaration))
            )

            if visible_boolean or visible_nonboolean:
                add(
                    "the controlling expression of an if statement and the controlling "
                    "expression of an iteration-statement shall have essentially Boolean type"
                )
                break

        return cues

    @staticmethod
    def _split_top_level_arguments(argument_text: str) -> list[str]:
        """Split one function-like macro invocation without evaluating C.

        This is intentionally a lightweight token-preserving helper. It only
        tracks nesting/strings well enough to recover the literal arguments
        needed for source-grounded macro substitution checks; it is not a C
        parser and never assigns a MISRA identifier.
        """

        text = str(argument_text or "")
        if not text.strip():
            return []

        parts: list[str] = []
        current: list[str] = []
        depth = 0
        quote = ""
        escape = False

        for char in text:
            if quote:
                current.append(char)
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == quote:
                    quote = ""
                continue

            if char in {'"', "'"}:
                quote = char
                current.append(char)
                continue
            if char in "([{":
                depth += 1
                current.append(char)
                continue
            if char in ")]}":
                depth = max(0, depth - 1)
                current.append(char)
                continue
            if char == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue
            current.append(char)

        parts.append("".join(current).strip())
        return parts

    @classmethod
    def _function_macro_definitions(cls, question: str) -> list[dict]:
        """Return visible function-like macro definitions and parameter protection.

        This helper is intentionally definition-based rather than invocation-based.
        It lets a fresh turn such as ``#define F(x) ((x) + 1)`` be assessed on
        its own without borrowing an older macro invocation from chat history.
        The analysis is conservative: directly parenthesized parameter occurrences
        are treated as protected; #/## operands and member-name occurrences are
        ignored because those uses do not by themselves form an expression.
        """

        raw = str(question or "")
        code = cls._code_only_view(raw) or raw
        definitions: list[dict] = []

        for macro in re.finditer(
            r"(?m)^\s*#\s*define\s+(?P<name>[A-Za-z_]\w*)\s*"
            r"\((?P<params>[^)]*)\)\s+(?P<body>[^\n]+)",
            code,
        ):
            name = macro.group("name")
            params = [item.strip() for item in macro.group("params").split(",") if item.strip()]
            body = macro.group("body").strip()
            if not params or not body:
                continue

            unsafe_params: list[str] = []
            expression_params: list[str] = []
            protected_params: list[str] = []

            for param in params:
                occurrences = list(
                    re.finditer(
                        rf"(?<![A-Za-z0-9_]){re.escape(param)}(?![A-Za-z0-9_])",
                        body,
                    )
                )
                if not occurrences:
                    continue

                param_has_expression_use = False
                param_unsafe = False
                param_protected = False

                for occ in occurrences:
                    left = body[:occ.start()]
                    right = body[occ.end():]
                    left_trim = left.rstrip()
                    right_trim = right.lstrip()

                    # Stringize/token-paste operands do not themselves form an
                    # expression under Rule 20.7.
                    if left_trim.endswith("##") or right_trim.startswith("##"):
                        continue
                    if left_trim.endswith("#") and not left_trim.endswith("##"):
                        continue

                    # A parameter used only as a member designator is not an
                    # expression occurrence by itself.
                    if left_trim.endswith(".") or left_trim.endswith("->"):
                        continue

                    param_has_expression_use = True
                    directly_parenthesized = (
                        left_trim.endswith("(") and right_trim.startswith(")")
                    )
                    if directly_parenthesized:
                        param_protected = True
                    else:
                        param_unsafe = True

                if param_has_expression_use:
                    expression_params.append(param)
                if param_unsafe:
                    unsafe_params.append(param)
                elif param_has_expression_use and param_protected:
                    protected_params.append(param)

            definitions.append({
                "name": name,
                "params": params,
                "body": body,
                "unsafe_params": unsafe_params,
                "expression_params": expression_params,
                "protected_params": protected_params,
            })

        return definitions

    @classmethod
    def _simple_function_macro_expansions(cls, question: str) -> list[dict]:
        """Return literal visible expansions for simple function-like macros.

        Only macro definitions and invocations that are visibly present in the
        user's text are considered. The expansion is a mechanical token
        substitution of parameters into the macro body; no operator folding,
        added parentheses, or outside compiler knowledge is introduced.
        """

        raw = str(question or "")
        code = cls._code_only_view(raw) or raw
        expansions: list[dict] = []

        for definition in cls._function_macro_definitions(raw):
            name = str(definition.get("name") or "")
            params = list(definition.get("params") or [])
            body = str(definition.get("body") or "")
            unsafe_params = list(definition.get("unsafe_params") or [])
            definition_match = re.search(
                rf"(?m)^\s*#\s*define\s+{re.escape(name)}\s*\([^)]*\)\s+[^\n]+",
                code,
            )
            if not definition_match:
                continue
            search_start = definition_match.end()
            call_re = re.compile(rf"\b{re.escape(name)}\s*\(")
            for call in call_re.finditer(code, search_start):
                open_index = code.find("(", call.start())
                if open_index < 0:
                    continue
                depth = 0
                quote = ""
                escape = False
                close_index = -1
                for index in range(open_index, len(code)):
                    char = code[index]
                    if quote:
                        if escape:
                            escape = False
                        elif char == "\\":
                            escape = True
                        elif char == quote:
                            quote = ""
                        continue
                    if char in {'"', "'"}:
                        quote = char
                        continue
                    if char == "(":
                        depth += 1
                    elif char == ")":
                        depth -= 1
                        if depth == 0:
                            close_index = index
                            break
                if close_index < 0:
                    continue

                args = cls._split_top_level_arguments(
                    code[open_index + 1:close_index]
                )
                if len(args) != len(params):
                    continue

                expanded = body
                for param, argument in zip(params, args):
                    expanded = re.sub(
                        rf"(?<![A-Za-z0-9_]){re.escape(param)}(?![A-Za-z0-9_])",
                        argument,
                        expanded,
                    )

                expansions.append({
                    "name": name,
                    "params": params,
                    "body": body,
                    "unsafe_params": unsafe_params,
                    "invocation": code[call.start():close_index + 1].strip(),
                    "expanded": re.sub(r"\s+", " ", expanded).strip(),
                })
                break

        return expansions

    @classmethod
    def cue_assessment_state(cls, question: str, cue: str) -> tuple[str, str]:
        """Return (state, observation) for a source-language cue.

        The state is derived from visible syntax/data flow, never from the user's
        wording. ``violation`` is the conservative default for historical cues.
        ``compliant`` is used only when the pasted construct visibly satisfies a
        source requirement/exception.
        """
        raw = str(question or "")
        code = cls._code_only_view(raw) or raw
        normalized = cls._normalized_phrase(cue)
        code_like = cls.looks_like_c_cpp(raw)

        # Source-grounded switch requirements need intent-aware assessment.
        # Handle them before the generic if/while Boolean branch so a natural
        # switch-expression question cannot be misclassified as an iteration
        # controlling-expression check.
        if normalized == cls._normalized_phrase(
            "a switch-expression shall not have essentially Boolean type"
        ):
            if not code_like:
                if re.search(r"\b(?:bool|boolean|essentially\s+boolean)\b", raw, re.IGNORECASE):
                    return (
                        "violation",
                        "The described switch expression is Boolean, which is exactly the type prohibited by the matched switch-expression requirement.",
                    )
                if re.search(r"\benum(?:eration|erated)?\b", raw, re.IGNORECASE):
                    return (
                        "uncertain",
                        "Rule 16.7 specifically addresses essentially Boolean switch expressions; an enum expression is not established as Boolean by the question alone, so other applicable MISRA rules must still be checked.",
                    )
                return (
                    "uncertain",
                    "The switch-expression requirement is relevant, but the question does not establish the expression's essential type.",
                )

            switch_expr = re.search(r"\bswitch\s*\(\s*(?P<expr>[^)]+?)\s*\)", code, re.IGNORECASE)
            if switch_expr:
                expr = re.sub(r"\s+", "", switch_expr.group("expr"))
                name = expr[:-2] if expr.endswith("()") else expr
                before = code[:switch_expr.start()]
                boolean_visible = bool(
                    re.search(rf"\b(?:bool|_Bool|bool_t)\s+{re.escape(name)}\b", before, re.IGNORECASE)
                    or re.search(rf"\b(?:bool|_Bool|bool_t)\s+{re.escape(name)}\s*\([^;{{}}]*\)\s*;", before, re.IGNORECASE)
                    or re.search(r"(?:==|!=|<=|>=|<|>|&&|\|\|)", switch_expr.group("expr"))
                )
                if boolean_visible:
                    return (
                        "violation",
                        f"The visible switch expression `{switch_expr.group('expr').strip()}` is Boolean or visibly produces a Boolean result.",
                    )
            if switch_expr:
                visible_expr = re.sub(r"\s+", " ", switch_expr.group("expr")).strip()
                return (
                    "uncertain",
                    f"The switch expression `{visible_expr}` is visible, but the snippet does not establish its essential type.",
                )
            return (
                "uncertain",
                "The matched Rule is relevant, but the snippet does not establish the switch expression's essential type.",
            )

        if normalized == cls._normalized_phrase(
            "every switch statement shall have a default label"
        ):
            if code_like and re.search(r"\bswitch\s*\(", code, re.IGNORECASE):
                if re.search(r"\bdefault\s*:", code, re.IGNORECASE):
                    return ("compliant", "The visible switch includes a default label.")
                return ("violation", "The visible switch has no default label.")
            if re.search(
                r"\b(?:with|has|have|includes?|contains?)\b.{0,28}"
                r"\b(?:(?:one|1|a)\s+)?default(?:\s+label|\s+case)?\b"
                r"|\b(?:one|1)\s+case\b.{0,40}\b(?:one|1|a)\s+default\b"
                r"|\bdefault\s+label\b.{0,24}\b(?:first|last|present|included)\b",
                raw,
                re.IGNORECASE,
            ):
                return (
                    "compliant",
                    "The described switch includes a default label, so this specific requirement is satisfied by the stated facts.",
                )
            return (
                "uncertain",
                "The default-label requirement is relevant, but the question does not establish whether a default label is present.",
            )

        if normalized == cls._normalized_phrase(
            "a default label shall appear as either the first or the last switch label of a switch statement"
        ):
            if code_like:
                labels = list(re.finditer(r"\b(?:case\b[^:]*|default)\s*:", code, re.IGNORECASE))
                default_positions = [i for i, label in enumerate(labels) if re.match(r"\s*default\b", label.group(0), re.IGNORECASE)]
                if default_positions:
                    pos = default_positions[0]
                    if pos == 0 and pos == len(labels) - 1:
                        return ("compliant", "The visible default label is the only switch label, so its position is both first and last.")
                    if pos == 0:
                        return ("compliant", "The visible default label is the first switch label.")
                    if pos == len(labels) - 1:
                        return ("compliant", "The visible default label is the last switch label.")
                    return ("violation", "The visible default label is between other switch labels instead of first or last.")
            prose_position = re.search(
                r"\bdefault(?:\s+label)?\b.{0,32}\b(?P<position>first|last)\b"
                r"|\b(?P<leading>first|last)\b.{0,32}\bdefault(?:\s+label)?\b",
                raw,
                re.IGNORECASE,
            )
            if prose_position:
                position = (prose_position.group("position") or prose_position.group("leading") or "").casefold()
                return (
                    "compliant",
                    f"The described default label is {position}, which is an allowed position under the matched requirement.",
                )
            return (
                "uncertain",
                "The default-label placement requirement is relevant, but its position is not established by the question.",
            )

        if normalized == cls._normalized_phrase(
            "every switch statement shall have at least two switch-clauses"
        ):
            labels = re.findall(r"\b(?:case\b[^:]*|default)\s*:", code, re.IGNORECASE) if code_like else []
            if labels:
                if len(labels) >= 2:
                    return (
                        "compliant",
                        f"The visible switch has {len(labels)} switch-clauses, satisfying the minimum of two.",
                    )
                return (
                    "violation",
                    f"The visible switch has only {len(labels)} switch-clause, below the minimum of two.",
                )
            prose = re.sub(r"\s+", " ", raw.casefold())
            if (
                re.search(r"\b(?:one|1)\s+case\b", prose)
                and re.search(r"\b(?:one|1|a)\s+default\b|\bdefault(?:\s+label|\s+case)?\b", prose)
            ) or re.search(r"\b(?:two|2)\s+switch[- ]clauses?\b", prose):
                return (
                    "compliant",
                    "The described switch has two switch-clauses, satisfying the minimum of two.",
                )
            return (
                "uncertain",
                "The minimum-clause requirement is relevant, but the number of switch-clauses is not established.",
            )

        # A text-only natural question can identify a strongly relevant MISRA
        # requirement without proving that a concrete program violates it.
        # Keep those answers useful but explicitly conditional.
        if (
            normalized == cls._normalized_phrase(
                "all exit paths from a function with non- void return type shall have an explicit return statement with an expression"
            )
            and not code_like
        ):
            return (
                "uncertain",
                "This requirement is directly relevant to the described missing-return path, but the function signature and complete control flow are needed to confirm an actual violation.",
            )

        if (
            "controlling expression" in normalized
            and "essentially boolean type" in normalized
        ):
            for control in re.finditer(
                r"\b(?P<keyword>if|while)\s*\(\s*(?P<neg>!\s*)?(?P<expr>[A-Za-z_]\w*(?:\s*\(\s*\))?)\s*\)",
                code,
                re.IGNORECASE,
            ):
                keyword = control.group("keyword")
                expr = re.sub(r"\s+", "", control.group("expr"))
                name = expr[:-2] if expr.endswith("()") else expr
                before = code[:control.start()]
                visible_text = f"{('!' if control.group('neg') else '')}{expr}"

                boolean_object = re.search(
                    rf"\b(?:bool|_Bool|bool_t)\s+{re.escape(name)}\b",
                    before,
                    re.IGNORECASE,
                )
                boolean_function = re.search(
                    rf"\b(?:bool|_Bool|bool_t)\s+{re.escape(name)}\s*\([^;{{}}]*\)\s*;",
                    before,
                    re.IGNORECASE,
                )
                pointer_object = re.search(
                    rf"\b(?:const\s+|volatile\s+|static\s+|extern\s+)*"
                    rf"(?:void|char|short|int|long|float|double|u?int\d+_t|[A-Za-z_]\w*_t)"
                    rf"\s*\*+\s*{re.escape(name)}\b",
                    before,
                    re.IGNORECASE,
                )
                scalar_object = re.search(
                    rf"\b(?:(?:signed|unsigned|const|volatile)\s+)*"
                    rf"(?:char|short|int|long|float|double|size_t|u?int\d+_t|intptr_t|uintptr_t)"
                    rf"\s+{re.escape(name)}\b",
                    before,
                    re.IGNORECASE,
                )
                pointer_function = re.search(
                    rf"\b(?:const\s+|volatile\s+|static\s+|extern\s+)*"
                    rf"(?:void|char|short|int|long|float|double|u?int\d+_t|[A-Za-z_]\w*_t)"
                    rf"\s*\*+\s*{re.escape(name)}\s*\([^;{{}}]*\)\s*;",
                    before,
                    re.IGNORECASE,
                )
                scalar_function = re.search(
                    rf"\b(?:(?:signed|unsigned|const|volatile)\s+)*"
                    rf"(?:char|short|int|long|float|double|size_t|u?int\d+_t|intptr_t|uintptr_t)"
                    rf"\s+{re.escape(name)}\s*\([^;{{}}]*\)\s*;",
                    before,
                    re.IGNORECASE,
                )

                if (expr.endswith("()") and boolean_function) or (not expr.endswith("()") and boolean_object):
                    return (
                        "compliant",
                        f"The visible {keyword} controlling expression `{visible_text}` is derived from a visibly declared Boolean value/function result, so this matched requirement is satisfied for that construct.",
                    )

                if (expr.endswith("()") and pointer_function) or (not expr.endswith("()") and pointer_object):
                    return (
                        "violation",
                        f"The visible {keyword} controlling expression `{visible_text}` is based on a pointer value, not an essentially Boolean value.",
                    )

                if (expr.endswith("()") and scalar_function) or (not expr.endswith("()") and scalar_object):
                    return (
                        "violation",
                        f"The visible {keyword} controlling expression `{visible_text}` is based on a non-Boolean scalar value, not an essentially Boolean value.",
                    )

            return (
                "uncertain",
                "The matched controlling-expression requirement is relevant, but the visible snippet does not establish the essential type of the controlling expression.",
            )

        if normalized == cls._normalized_phrase(
            "the macro NULL shall be the only permitted form of integer null pointer constant"
        ):
            match = re.search(
                r"\b(?:const\s+|volatile\s+|static\s+|extern\s+)*"
                r"(?:void|char|short|int|long|float|double|u?int\d+_t|[A-Za-z_]\w*_t)"
                r"\s*\*+\s*(?P<name>[A-Za-z_]\w*)\s*=\s*0(?:[uUlL]*)\s*;",
                code,
                re.IGNORECASE,
            )
            if match:
                return (
                    "violation",
                    f"The visible pointer initialization `{match.group('name')} = 0` uses the integer constant 0 instead of the macro NULL.",
                )
            return ("violation", cue.rstrip(" .") + ".")

        if normalized == cls._normalized_phrase(
            "the body of an iteration-statement or a selection-statement shall be a compound-statement"
        ):
            unbraced = re.search(
                r"\b(?P<keyword>if|for|while)\s*\([^\n]*?\)\s*(?!\{)(?P<body>[^\n{{}};]+(?:;|$))",
                code,
                re.IGNORECASE,
            )
            if unbraced:
                return (
                    "violation",
                    f"The visible {unbraced.group('keyword')} body is a single unbraced statement rather than a compound statement enclosed in braces.",
                )
            return ("violation", cue.rstrip(" .") + ".")

        if normalized == cls._normalized_phrase(
            "the right hand operand of a logical && or || operator shall not contain persistent side effects"
        ):
            for logical in re.finditer(r"(?:&&|\|\|)(?P<rhs>[^;\n]+)", code):
                rhs = logical.group("rhs")
                if re.search(r"(?<![=!<>])=(?!=)", rhs) or "++" in rhs or "--" in rhs:
                    return (
                        "violation",
                        "The visible right-hand operand contains an assignment or increment/decrement, which is a persistent side effect.",
                    )
                if re.search(r"\b(?!(?:if|for|while|switch|sizeof)\b)[A-Za-z_]\w*\s*\(", rhs, re.I):
                    return (
                        "uncertain",
                        "A function call appears in the right-hand operand, but the visible snippet does not establish whether that call has a persistent side effect.",
                    )
            if not cls.looks_like_c_cpp(question):
                return (
                    "uncertain",
                    "The matched requirement is relevant, but the question alone does not establish whether the function call has a persistent side effect in the right-hand operand.",
                )
            return (
                "uncertain",
                "The matched requirement is relevant, but the visible snippet is not sufficient to confirm a persistent side effect in the right-hand operand.",
            )

        if normalized == cls._normalized_phrase(
            "the +, -, += and -= operators should not be applied to an expression of pointer type"
        ):
            pointer_names = {
                match.group("name")
                for match in re.finditer(
                    r"\b(?:const\s+|volatile\s+|static\s+|extern\s+)*"
                    r"(?:void|char|short|int|long|float|double|u?int\d+_t|[A-Za-z_]\w*_t)"
                    r"\s*\*+\s*(?P<name>[A-Za-z_]\w*)\b",
                    code,
                    re.IGNORECASE,
                )
            }
            for raw_line in code.splitlines():
                line = re.sub(r"\s+", " ", raw_line).strip()
                if not line:
                    continue
                for pointer_name in pointer_names:
                    operator_match = re.search(
                        rf"\b{re.escape(pointer_name)}\b\s*(?P<operator>\+=|-=|\+|-)\s*[^;]+",
                        line,
                    )
                    if operator_match:
                        operator = operator_match.group("operator")
                        return (
                            "violation",
                            f"The visible pointer-arithmetic construct `{line}` uses the `{operator}` operator with the pointer `{pointer_name}`.",
                        )
            return ("violation", cue.rstrip(" .") + ".")

        if normalized == cls._normalized_phrase(
            "expressions resulting from the expansion of macro parameters shall be enclosed in parentheses"
        ):
            definitions = cls._function_macro_definitions(raw)
            expansions = cls._simple_function_macro_expansions(raw)

            # A visible function-like macro definition is sufficient to assess
            # direct parameter protection.  This is deliberately independent
            # of any older invocation in chat history, so a fresh corrected
            # macro replaces the prior unsafe snippet as the active scenario.
            for definition in definitions:
                unsafe_params = list(definition.get("unsafe_params") or [])
                if not unsafe_params:
                    continue
                parameter_text = ", ".join(f"`{value}`" for value in unsafe_params)
                detail = (
                    f"The macro parameter(s) {parameter_text} are used as expressions without enclosing parentheses in the visible macro body."
                )
                matching_expansion = next(
                    (item for item in expansions if item.get("name") == definition.get("name")),
                    None,
                )
                if matching_expansion:
                    invocation = str(matching_expansion.get("invocation") or "").strip()
                    expanded = str(matching_expansion.get("expanded") or "").strip()
                    if invocation and expanded:
                        detail += (
                            f" Literal substitution of `{invocation}` produces `{expanded}`; the expansion must not be rewritten with parentheses that are absent from the shown definition/arguments."
                        )
                return ("violation", detail)

            protected = [
                item for item in definitions
                if item.get("expression_params") and not item.get("unsafe_params")
            ]
            if protected:
                return (
                    "compliant",
                    "The visible function-like macro parameter occurrences used as expressions are enclosed in parentheses in the shown macro definition.",
                )

            return ("uncertain", "The visible information does not establish a macro-parameter expression that can be assessed against this requirement.")

        if normalized == cls._normalized_phrase(
            "a pointer resulting from arithmetic on a pointer operand shall address an element of the same array as that pointer operand"
        ):
            # Known out-of-range array subscript -> potential violation.
            declaration_matches = list(re.finditer(
                r"\b(?:char|short|int|long|float|double|u?int\d+_t|[A-Za-z_]\w*_t)"
                r"\s+(?P<name>[A-Za-z_]\w*)\s*\[\s*(?P<size>\d+)\s*\]",
                code, re.I,
            ))
            array_sizes = {m.group("name"): int(m.group("size")) for m in declaration_matches}
            decl_spans = [(m.start(), m.end()) for m in declaration_matches]
            integer_values = {
                m.group("name"): int(m.group("value"))
                for m in re.finditer(
                    r"\b(?:signed\s+|unsigned\s+)?(?:char|short|int|long|u?int\d+_t|size_t)"
                    r"\s+(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>-?\d+)[uUlL]*\s*;",
                    code, re.I,
                )
            }
            for access in re.finditer(r"\b(?P<array>[A-Za-z_]\w*)\s*\[\s*(?P<index>-?\d+|[A-Za-z_]\w*)\s*\]", code):
                if any(start <= access.start() < end for start, end in decl_spans):
                    continue
                array_name = access.group("array")
                if array_name not in array_sizes:
                    continue
                token = access.group("index")
                try:
                    index_value = int(token)
                except ValueError:
                    index_value = integer_values.get(token)
                if index_value is not None and not (0 <= index_value < array_sizes[array_name]):
                    return (
                        "violation",
                        f"The visible array index {index_value} is outside the valid range for {array_name}[{array_sizes[array_name]}].",
                    )

            # Known array-backed pointer with an in-range constant offset.
            for base in re.finditer(
                r"\b(?:const\s+|volatile\s+|static\s+|extern\s+)*"
                r"(?:void|char|short|int|long|float|double|u?int\d+_t|[A-Za-z_]\w*_t)"
                r"\s*\*+\s*(?P<pointer>[A-Za-z_]\w*)\s*=\s*(?P<array>[A-Za-z_]\w*)\s*;",
                code, re.I,
            ):
                pointer_name = base.group("pointer")
                array_name = base.group("array")
                if array_name not in array_sizes:
                    continue
                arithmetic = re.search(
                    rf"\b{re.escape(pointer_name)}\s*=\s*{re.escape(pointer_name)}\s*(?P<op>[+-])\s*(?P<offset>\d+)\s*;",
                    code, re.I,
                )
                if not arithmetic:
                    continue
                offset = int(arithmetic.group("offset"))
                if arithmetic.group("op") == "-":
                    offset = -offset
                if 0 <= offset <= array_sizes[array_name]:
                    return (
                        "compliant",
                        f"The visible pointer starts from {array_name} and moves by {offset} element(s), remaining within the same {array_sizes[array_name]}-element array for pointer creation.",
                    )
            return (
                "uncertain",
                "The same-array requirement is relevant to the pointer arithmetic, but the visible snippet does not prove the resulting pointer's bounds.",
            )

        if normalized == cls._normalized_phrase(
            "the value of an object with automatic storage duration shall not be read before it has been set"
        ):
            # File-scope scalar declarations have static storage duration; the
            # Rule 9.1 rationale explicitly states they are zero-initialized.
            function = re.search(
                r"\b(?:[A-Za-z_]\w*\s+)+[A-Za-z_]\w*\s*\([^;{}]*\)\s*\{", code
            )
            prefix = code[:function.start()] if function else code
            if re.search(
                r"(?m)^\s*(?:signed\s+|unsigned\s+)?(?:char|short|int|long|float|double|u?int\d+_t)"
                r"\s+[A-Za-z_]\w*\s*;", prefix
            ):
                return (
                    "compliant",
                    "The questioned object is declared at file scope; Rule 9.1 is limited to objects with automatic storage duration, and its rationale states that static-storage objects are automatically initialized to zero.",
                )
            return (
                "violation",
                "A visible automatic object can be read on a path where no prior guaranteed assignment is shown.",
            )

        if normalized == cls._normalized_phrase(
            "a full expression containing an increment or decrement operator should have no other potential side effects other than that caused by the increment or decrement operator"
        ):
            # If the increment/decrement appears in a controlling expression but
            # no other side effect is visible in that same full expression, the
            # source requirement is satisfied.
            for condition in re.finditer(r"\b(?:if|while)\s*\((?P<expr>.*?)\)\s*\{", code, re.I | re.S):
                expr = condition.group("expr")
                if "++" not in expr and "--" not in expr:
                    continue
                remainder = re.sub(r"\+\+|--", "", expr)
                other_assignment = bool(re.search(r"(?<![=!<>])=(?!=)", remainder))
                other_call = bool(re.search(r"\b[A-Za-z_]\w*\s*\(", remainder))
                if not other_assignment and not other_call:
                    return (
                        "compliant",
                        "The visible full expression contains the increment/decrement side effect but no additional assignment or function-call side effect.",
                    )
            return ("violation", cue.rstrip(" .") + ".")

        if normalized == cls._normalized_phrase("a for loop shall be well-formed") and re.search(
            r"\bfor\s*\(\s*;\s*;\s*\)", code, re.I
        ):
            return (
                "compliant",
                "The source exception to Rule 14.2 explicitly permits all three for-loop clauses to be empty for an infinite loop.",
            )

        if normalized == cls._normalized_phrase(
            "there should be no more than one break or goto statement used to terminate any iteration statement"
        ):
            exits = len(re.findall(r"\b(?:break|goto\s+[A-Za-z_]\w*)\s*;", code, re.I))
            if exits <= 1:
                return (
                    "compliant",
                    "Only one visible break/goto is used for early loop termination, which satisfies the matched advisory requirement.",
                )

        if normalized == cls._normalized_phrase("every switch statement shall have at least two switch-clauses"):
            labels = re.findall(r"\b(?:case\b[^:]*|default)\s*:", code, re.I)
            if len(labels) >= 2:
                return (
                    "compliant",
                    f"The visible switch has {len(labels)} switch-clauses, satisfying the minimum of two.",
                )

        return ("violation", cue.rstrip(" .") + ".")

    @classmethod
    def build_search_query(cls, question: str, resolved_question: str = "") -> str:
        raw = str(question or "").strip()
        resolved = str(resolved_question or "").strip()

        # v6.4.26 prefixed every request with generic words such as
        # "rule/directive/rationale/example".  On the real MISRA corpus those
        # terms over-selected indexes/appendices and hid the actual rule body.
        # Keep only a small source anchor, then add construct-specific semantic
        # cues derived from the user's own code/scenario. No Rule/Directive
        # number is inserted here.
        parts = ["MISRA C"]
        parts.extend(cls.semantic_cues(raw))
        parts.append(raw)

        if resolved and resolved.casefold() != raw.casefold():
            parts.append(resolved)

        return " | ".join(part for part in parts if part)[:5000]

    @staticmethod
    def _lexical_tokens(text: str) -> list[str]:
        """Return conservative tokens for deterministic MISRA cue matching."""
        stop = {
            "a", "an", "the", "of", "to", "for", "in", "on", "at", "and", "or",
            "with", "from", "by", "as", "is", "are", "was", "were", "be", "been",
            "this", "that", "these", "those", "it", "its", "should", "shall", "not",
        }
        output: list[str] = []
        seen = set()
        for token in re.findall(r"[a-z0-9]+", str(text or "").casefold()):
            if len(token) <= 2 or token in stop or token in seen:
                continue
            seen.add(token)
            output.append(token)
        return output

    @staticmethod
    def _normalized_phrase(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", str(text or "").casefold()))

    @classmethod
    def _is_citable_rule_body_record(cls, item: Mapping) -> bool:
        metadata = item.get("metadata", {}) if isinstance(item, Mapping) else {}
        if not cls.result_is_misra(item):
            return False
        section_type = str(metadata.get("section_type", "") or "").casefold()
        if section_type == "rule":
            return bool(str(metadata.get("rule_id", "") or "").strip())
        if section_type == "directive":
            return bool(
                str(metadata.get("directive_id", "") or metadata.get("rule_id", "") or "").strip()
            )
        return False

    @staticmethod
    def _reference_key(item: Mapping) -> tuple[str, str]:
        metadata = item.get("metadata", {}) if isinstance(item, Mapping) else {}
        section_type = str(metadata.get("section_type", "") or "").casefold()
        if section_type == "directive":
            identifier = str(
                metadata.get("directive_id", "") or metadata.get("rule_id", "") or ""
            ).strip()
            return ("directive", identifier)
        return ("rule", str(metadata.get("rule_id", "") or "").strip())

    @classmethod
    def _expand_rule_body_details(
        cls,
        records: Sequence[Mapping],
        selected: Mapping,
    ) -> Mapping:
        """Attach only detail sections that belong immediately after a rule body.

        Structure-aware MISRA ingestion stores the Rule/Directive heading as one
        chunk and labels such as Amplification, Rationale, Exception and Example
        as following section chunks.  Keeping those sections with the selected
        rule gives answer generation the same meaningful structure as the source
        without borrowing text from a neighboring rule or appendix.
        """
        metadata = dict(selected.get("metadata", {}) or {})
        file_name = str(metadata.get("file_name", "") or "")
        try:
            start_chunk = int(metadata.get("chunk_id"))
        except (TypeError, ValueError):
            return dict(selected)

        allowed_details = {
            "amplification", "rationale", "exception", "exceptions",
            "example", "examples", "note", "notes", "see also",
        }
        pieces = [str(selected.get("text", "") or "").strip()]
        page_end = metadata.get("page_end") or metadata.get("page_start")

        by_chunk = {}
        for record in records or []:
            record_meta = record.get("metadata", {}) if isinstance(record, Mapping) else {}
            if str(record_meta.get("file_name", "") or "") != file_name:
                continue
            try:
                chunk_id = int(record_meta.get("chunk_id"))
            except (TypeError, ValueError):
                continue
            by_chunk[chunk_id] = record

        next_chunk = start_chunk + 1
        while next_chunk in by_chunk:
            record = by_chunk[next_chunk]
            record_meta = record.get("metadata", {}) or {}
            section_type = str(record_meta.get("section_type", "") or "").casefold()
            if section_type in {"rule", "directive"}:
                break
            title = str(record_meta.get("section_title", "") or "").strip()
            title_key = re.sub(r"\s+", " ", title.casefold()).strip()
            if title_key not in allowed_details:
                break
            text = str(record.get("text", "") or "").strip()
            if text:
                pieces.append(text)
            page_end = record_meta.get("page_end") or record_meta.get("page_start") or page_end
            next_chunk += 1

        enriched = dict(selected)
        enriched["metadata"] = metadata
        enriched["text"] = "\n\n".join(piece for piece in pieces if piece)
        if page_end:
            enriched["metadata"]["page_end"] = page_end
        enriched["_misra_rule_body_rescue"] = True
        return enriched

    @classmethod
    def load_authoritative_bm25_records(cls) -> list[Mapping]:
        """Load the on-disk BM25 corpus used by the running project.

        Natural MISRA rescue must not depend on a possibly stale process-local
        Streamlit resource wrapper.  The normal retriever still owns all general
        search behaviour; this narrow helper only reads the same authoritative
        ``storage/bm25/corpus.pkl`` that BM25Searcher uses.
        """

        corpus_path = BM25_DIR / "corpus.pkl"
        if not corpus_path.exists():
            return []

        try:
            with open(corpus_path, "rb") as file:
                records = pickle.load(file)
        except Exception:
            return []

        if not isinstance(records, list):
            return []
        return [item for item in records if isinstance(item, Mapping)]

    @classmethod
    def rule_body_cue_rescue_from_authoritative_corpus(
        cls,
        question: str,
        top_k: int = 6,
    ) -> tuple[list[Mapping], dict]:
        """Run the MISRA-only rescue directly against the authoritative corpus."""

        records = cls.load_authoritative_bm25_records()
        citable_count = sum(
            1 for item in records if cls._is_citable_rule_body_record(item)
        )
        rescued = cls.rule_body_cue_rescue(
            records=records,
            question=question,
            top_k=top_k,
        )
        diagnostics = {
            "record_count": len(records),
            "citable_rule_body_count": citable_count,
            "semantic_cues": cls.semantic_cues(question),
            "matched_references": [
                str((item.get("metadata", {}) or {}).get("section_title", ""))
                for item in rescued
            ],
        }
        return rescued, diagnostics

    @classmethod
    def rule_body_cue_rescue(
        cls,
        records: Sequence[Mapping],
        question: str,
        top_k: int = 6,
    ) -> list[Mapping]:
        """Recover strongly matching actual MISRA Rule/Directive body chunks.

        This rescue is intentionally narrower than the global retriever.  It is
        used only in natural MISRA-compliance mode and does not lower the global
        reranker threshold.  No Rule/Directive number is mapped from a canned
        question: visible code constructs create source-language cues, and a
        candidate is eligible only when an actual structured Rule/Directive body
        lexically covers one of those cues strongly enough.
        """
        cues = cls.semantic_cues(question)
        if not cues or not records:
            return []

        citable_records = [
            item for item in records
            if isinstance(item, Mapping) and cls._is_citable_rule_body_record(item)
        ]
        chosen: dict[tuple[str, str], Mapping] = {}

        for cue_index, cue in enumerate(cues):
            cue_tokens = cls._lexical_tokens(cue)
            if len(cue_tokens) < 2:
                continue
            cue_set = set(cue_tokens)
            cue_phrase = cls._normalized_phrase(cue)
            best = None
            best_rank = (-1.0, -1, -1)

            for item in citable_records:
                text = str(item.get("text", "") or "")
                body_tokens = set(cls._lexical_tokens(text))
                matched = cue_set.intersection(body_tokens)
                coverage = len(matched) / max(1, len(cue_set))
                phrase_match = bool(
                    cue_phrase and cue_phrase in cls._normalized_phrase(text)
                )
                minimum_matches = max(2, int(round(len(cue_set) * 0.60)))

                if not phrase_match and (
                    len(matched) < minimum_matches or coverage < 0.72
                ):
                    continue

                rank = (
                    1.0 if phrase_match else coverage,
                    len(matched),
                    -len(body_tokens),
                )
                if rank > best_rank:
                    best_rank = rank
                    best = item

            if best is None:
                continue

            key = cls._reference_key(best)
            if not key[1]:
                continue
            enriched = cls._expand_rule_body_details(records, best)
            enriched = dict(enriched)
            enriched["score"] = max(
                float(enriched.get("score", 0.0) or 0.0),
                1.0 if best_rank[0] >= 1.0 else float(best_rank[0]),
            )
            enriched["_misra_cue_index"] = cue_index
            enriched["_misra_cue"] = cue
            enriched["_misra_cue_coverage"] = float(best_rank[0])
            state, observation = cls.cue_assessment_state(question, cue)
            enriched["_misra_assessment_state"] = state
            enriched["_misra_observation"] = observation
            chosen[key] = enriched

        results = list(chosen.values())
        results.sort(
            key=lambda item: (
                int(item.get("_misra_cue_index", 999)),
                -float(item.get("_misra_cue_coverage", 0.0) or 0.0),
            )
        )
        return results[: max(1, int(top_k))]

    @classmethod
    def exact_reference_evidence(
        cls,
        records: Sequence[Mapping],
        question: str,
    ) -> list[Mapping]:
        """Return authoritative Rule/Directive anchors named by the current turn.

        Explanation, rationale, and example requests need the parent Rule body as
        the citable anchor even when a child section ranks higher semantically.
        This planner resolves only identifiers explicitly written by the user and
        attaches same-rule detail chunks through the existing structure-aware
        expansion helper.
        """
        refs = [
            ref
            for ref in extract_structured_references(str(question or ""))
            if ref.kind in {"rule", "directive"}
        ]
        if not refs or not records:
            return []
        wanted = []
        for ref in refs:
            key = (ref.kind, ref.identifier)
            if ref.identifier and key not in wanted:
                wanted.append(key)

        out = []
        for kind, identifier in wanted:
            selected = None
            for item in records:
                if not isinstance(item, Mapping) or not cls._is_citable_rule_body_record(item):
                    continue
                item_kind, item_id = cls._reference_key(item)
                if item_kind == kind and item_id == identifier:
                    selected = item
                    break
            if selected is None:
                continue
            enriched = dict(cls._expand_rule_body_details(records, selected))
            enriched["_misra_exact_reference_anchor"] = True
            enriched["_exact_structured_match"] = True
            enriched["metadata"]["exact_structured_match"] = True
            enriched["_misra_rule_body_rescue"] = True
            enriched["_misra_cue"] = cls._source_rule_statement(str(enriched.get("text", "") or ""))
            enriched["_misra_cue_coverage"] = 1.0
            enriched["_misra_assessment_state"] = "uncertain"
            enriched["_misra_observation"] = ""
            out.append(enriched)
        return out

    @classmethod
    def deterministic_followup_detail(
        cls,
        question: str,
        results: Sequence[Mapping],
    ) -> str:
        """Answer narrow grounded MISRA follow-ups directly from source evidence.

        This is intentionally relation-driven rather than Rule-number-driven. It
        handles two high-value conversational cases where a generative rewrite can
        lose an already-proven fact: asking *why* a matched requirement exists, and
        asking whether a visibly satisfied construct still needs a change.
        """

        clean = re.sub(r"\s+", " ", str(question or "").strip().casefold())
        if not clean:
            return ""

        candidates = [
            item for item in (results or [])
            if isinstance(item, Mapping)
            and (
                item.get("_misra_rule_body_rescue") is True
                or item.get("_exact_structured_match") is True
                or (
                    bool(item.get("_structured_topic_family"))
                    and "_misra_assessment_state" in item
                )
            )
            and cls._is_citable_rule_body_record(item)
        ]
        if not candidates:
            return ""

        # A visible function-like macro definition/invocation carries a
        # mechanically derivable literal expansion. Even when the wording is
        # "explain why", do not collapse the response to rationale-only prose;
        # let the full deterministic assessment preserve the Rule identifier,
        # status, visible observation, and exact literal substitution.
        if cls._simple_function_macro_expansions(str(question or "")):
            return ""

        asks_why = bool(re.match(
            r"^(?:why\b|bakit\b|(?:explain|ipaliwanag|paliwanag).{0,35}\b(?:why|bakit|restricted|reason)\b|"
            r"ano(?:ng)?\s+(?:reason|rationale)\b)",
            clean,
            re.IGNORECASE,
        ))
        asks_change = bool(re.search(
            r"(?:\b(?:change|fix|modify|correct|baguhin|ayusin)\b|"
            r"\bkailangan\b.{0,24}\b(?:baguhin|ayusin|change|fix)\b)",
            clean,
            re.IGNORECASE,
        ))

        if asks_why and not re.search(
            r"\b(?:which\s+part|aling\s+part|exact\s+part|basis|code|snippet|line)\b",
            clean,
            re.IGNORECASE,
        ):
            rendered: list[str] = []
            for item in candidates:
                metadata = item.get("metadata", {}) or {}
                kind = "Directive" if str(metadata.get("section_type", "")).casefold() == "directive" else "Rule"
                identifier = str(metadata.get("directive_id", "") or metadata.get("rule_id", "") or "").strip()
                reference = f"{kind} {identifier}".strip()
                rationale = cls._source_labeled_block(str(item.get("text", "") or ""), "Rationale")
                # Keep the complete compact rationale through the source's concluding
                # recommendation. Rule 16.7, for example, explains the Boolean/integer
                # relationship in two sentences and gives the important if-else
                # alternative in the third sentence.
                rationale = cls._first_sentences(rationale, maximum=3, max_chars=700)
                if rationale:
                    rendered.append(f"- **{reference}:** {rationale}" if len(candidates) > 1 else f"**{reference}:** {rationale}")
            if rendered:
                return "\n".join(rendered).strip()

        asks_example = bool(re.search(
            r"\b(?:example|examples|sample|illustrat(?:e|ion)|practical)\b",
            clean,
            re.IGNORECASE,
        ))
        asks_explain = bool(re.match(r"^(?:explain|describe|summari[sz]e)\b", clean))
        asks_simple = bool(re.search(r"\b(?:simple\s+terms?|plain\s+english|simply)\b", clean))
        if asks_explain and not asks_why and not asks_example:
            rendered: list[str] = []
            for item in candidates:
                metadata = item.get("metadata", {}) or {}
                kind = "Directive" if str(metadata.get("section_type", "")).casefold() == "directive" else "Rule"
                identifier = str(metadata.get("directive_id", "") or metadata.get("rule_id", "") or "").strip()
                reference = f"{kind} {identifier}".strip()
                statement = cls._source_rule_statement(str(item.get("text", "") or ""))
                if not statement:
                    continue
                lines = [f"**{reference}:** {statement}"]
                if asks_simple:
                    plain = statement.rstrip(" .")
                    plain = re.sub(r"\bshall\s+not\b", "do not", plain, flags=re.IGNORECASE)
                    plain = re.sub(r"\bshould\s+not\b", "avoid", plain, flags=re.IGNORECASE)
                    plain = re.sub(r"\bshall\b", "must", plain, flags=re.IGNORECASE)
                    plain = re.sub(r"\bswitch-expression\b", "switch expression", plain, flags=re.IGNORECASE)
                    plain = re.sub(r"\bessentially\s+Boolean\s+type\b", "a Boolean-type value", plain, flags=re.IGNORECASE)
                    if re.search(r"switch\s+expression", plain, re.IGNORECASE) and re.search(r"Boolean", plain, re.IGNORECASE):
                        plain = "Avoid using a Boolean-type value as the switch expression"
                    if plain and plain.casefold() != statement.casefold():
                        lines.append(f"In simple terms: {plain[0].upper() + plain[1:]}.")
                rationale = cls._source_labeled_block(str(item.get("text", "") or ""), "Rationale")
                rationale = cls._first_sentences(rationale, maximum=1, max_chars=320)
                if rationale and not asks_simple:
                    lines.append(f"Why: {rationale}")
                rendered.append("\n\n".join(lines))
            if rendered:
                return "\n\n".join(rendered).strip()

        if asks_example:
            rendered: list[str] = []
            for item in candidates:
                metadata = item.get("metadata", {}) or {}
                kind = "Directive" if str(metadata.get("section_type", "")).casefold() == "directive" else "Rule"
                identifier = str(metadata.get("directive_id", "") or metadata.get("rule_id", "") or "").strip()
                reference = f"{kind} {identifier}".strip()
                statement = cls._source_rule_statement(str(item.get("text", "") or ""))
                example = cls._source_labeled_block(str(item.get("text", "") or ""), "Example")
                if not example:
                    example = cls._source_labeled_block(str(item.get("text", "") or ""), "Examples")
                example = cls._first_sentences(example, maximum=3, max_chars=900) or re.sub(r"\s+", " ", example).strip()
                if example:
                    intro = f"**{reference}**"
                    if statement:
                        intro += f" — {statement}"
                    rendered.append(f"{intro}\n\nExample: {example}")
            if rendered:
                return "\n\n".join(rendered).strip()

        if asks_change:
            compliant = [
                item for item in candidates
                if str(item.get("_misra_assessment_state", "") or "").casefold() == "compliant"
            ]
            if compliant and len(compliant) == len(candidates):
                details: list[str] = []
                for item in compliant:
                    metadata = item.get("metadata", {}) or {}
                    kind = "Directive" if str(metadata.get("section_type", "")).casefold() == "directive" else "Rule"
                    identifier = str(metadata.get("directive_id", "") or metadata.get("rule_id", "") or "").strip()
                    reference = f"{kind} {identifier}".strip()
                    observation = re.sub(r"\s+", " ", str(item.get("_misra_observation", "") or "")).strip()
                    if observation:
                        details.append(f"{reference}: {observation}")
                if details:
                    return "No. " + " ".join(details)

        return ""


    @classmethod
    def deterministic_exact_rule_application(
        cls,
        question: str,
        results: Sequence[Mapping],
    ) -> str:
        """Apply a simple numeric minimum from an exact Rule body safely.

        This intentionally handles only mechanically decidable wording such as
        "at least N <items>" when the user's question explicitly supplies the
        corresponding count.  It prevents an exact, fully grounded Rule lookup
        from being sent to an LLM that may incorrectly fall back.
        """

        question_text = re.sub(r"\s+", " ", str(question or "")).strip()
        reference_match = _REF_RE.search(question_text)
        if not reference_match:
            return ""

        requested_id = reference_match.group("identifier")
        exact = None
        for item in results or []:
            if not isinstance(item, Mapping):
                continue
            metadata = item.get("metadata", {}) or {}
            rule_id = str(metadata.get("rule_id", "") or "").strip()
            if rule_id == requested_id and (
                item.get("_exact_structured_match")
                or metadata.get("exact_structured_match")
            ):
                exact = item
                break

        if exact is None:
            return ""

        statement = cls._source_rule_statement(str(exact.get("text", "") or ""))
        statement_clean = re.sub(r"\s+", " ", statement).strip()
        if not statement_clean:
            return ""

        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        minimum_match = re.search(
            r"\bat\s+least\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+([a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,2})",
            statement_clean.casefold(),
        )
        if not minimum_match:
            return ""

        minimum_token = minimum_match.group(1)
        minimum = int(minimum_token) if minimum_token.isdigit() else number_words[minimum_token]
        item_phrase = minimum_match.group(2).strip()
        item_tokens = [token for token in re.findall(r"[a-z]+", item_phrase) if len(token) > 2]
        if not item_tokens:
            return ""

        count_noun = item_tokens[-1]
        if count_noun.endswith("s") and len(count_noun) > 3:
            count_noun = count_noun[:-1]
        count_pattern = (
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b"
            r"[^.!?\n]{0,35}\b" + re.escape(count_noun) + r"s?\b"
        )
        count_match = re.search(count_pattern, question_text.casefold())
        if not count_match:
            return ""

        count_token = count_match.group(1)
        actual = int(count_token) if count_token.isdigit() else number_words[count_token]
        satisfied = actual >= minimum
        reference = f"Rule {requested_id}"

        if satisfied:
            assessment = (
                f"Yes. Based on the stated minimum in {reference}, the described "
                f"count satisfies this requirement."
            )
            result_line = f"{actual} meets the minimum of {minimum}."
        else:
            assessment = (
                f"No. Based on the stated minimum in {reference}, the described "
                f"count does not satisfy this requirement."
            )
            result_line = f"{actual} is below the minimum of {minimum}."

        return (
            f"{assessment}\n\n"
            f"- **Requirement:** {statement_clean.rstrip('.')} .\n"
            f"- **Result:** {result_line}"
        ).replace(" .", ".")

    @classmethod
    def should_use_llm_generation(
        cls,
        question: str,
        results: Sequence[Mapping],
        *,
        grounded_followup: bool = False,
    ) -> bool:
        """Return True when a grounded MISRA answer benefits from the complex LLM.

        Structured retrieval remains deterministic and authoritative.  The LLM is
        used only to explain/synthesize already-grounded evidence for natural
        reviewer-style questions, multi-rule cases, uncertain assessments, or
        grounded follow-ups.  Simple direct checks keep the certified low-latency
        deterministic finalizer.
        """

        clean = re.sub(r"\s+", " ", str(question or "").strip().casefold())
        candidates = [item for item in (results or []) if isinstance(item, Mapping)]
        states = {str(item.get("_misra_assessment_state", "") or "").casefold() for item in candidates}

        # Text-only concept/requirement questions are safest and fastest when
        # finalized from the already matched Rule/Directive body.  Do not ask a
        # model to invent a "user code" snippet merely to explain a concept.
        if not cls.looks_like_c_cpp(question) and cls.semantic_cues(question):
            return False

        # A grounded follow-up with no dedicated deterministic detail can still
        # benefit from synthesis.  The narrow why/change follow-ups are handled
        # by ``deterministic_followup_detail`` before this decision.
        if grounded_followup:
            return True

        # Direct code assessments whose visible states are already deterministic
        # do not need a second model pass.  Keep reviewer-style narrative on the
        # complex model, but make direct safe/compliant/concern checks fast and
        # exact even for mixed/uncertain states.
        direct_assessment = bool(re.search(
            r"^(?:safe\s+ba|compliant\s+ba|acceptable\s+ba|okay\s+ba|ok\s+ba|"
            r"is\s+(?:this|it)\s+(?:safe|compliant|acceptable)|"
            r"may\s+(?:visible\s+)?(?:misra\s+)?(?:violation|concern|issue))\b",
            clean,
            re.IGNORECASE,
        ))
        if direct_assessment and candidates:
            return False

        # Literal function-like macro expansion is mechanically derivable from
        # the user's exact definition/invocation.  When the user asks to explain
        # that expansion, deterministic finalization is both safer and faster.
        if cls._simple_function_macro_expansions(str(question or "")):
            return False

        rich_intent = re.search(
            r"\b(?:review|reviewer|explain|describe|why|bakit|paano|how|"
            r"fix|aayusin|safest|direction|context|confirm|confirmed|"
            r"multiple|more than one|iba pa|relevant rule|satisfied|"
            r"naturally|developer|risk|concern(?:s)?)\b",
            clean,
            re.IGNORECASE,
        )
        if rich_intent:
            return True

        # Keep a low-latency deterministic safety path for terse direct checks,
        # even when more than one Rule is source-proven. Rich reviewer-style or
        # uncertain multi-rule questions above still go to the complex model.
        simple_direct = re.match(
            r"(?i)^(?:compliant\s+ba|is\s+(?:this|it)\s+compliant|"
            r"misra\s+check(?:\s+please)?|okay\s+ba\s+(?:ito\s+)?(?:kay\s+)?misra|"
            r"acceptable\s+ba\s+(?:ito\s+)?(?:kay\s+)?misra)",
            clean,
        )
        if len(cls.available_references(candidates)) >= 2 and not simple_direct:
            return True
        return False

    @classmethod
    def deterministic_code_review_fast_path(
        cls,
        question: str,
        results: Sequence[Mapping],
    ) -> tuple[str, dict]:
        """Return a self-verified deterministic code-review answer when safe.

        Phase 7 targets a measured waste pattern: some natural reviewer-style
        MISRA questions spend 10-30 seconds on Qwen synthesis even though the
        visible-code analyzer has already assigned every relevant Rule a
        deterministic state and the semantic guard ultimately accepts (or
        falls back to) the same deterministic assessment.

        This helper does *not* relax retrieval, confidence, or semantic rules.
        It builds the existing deterministic assessment and then runs that
        answer through the same reference-grounding and semantic-alignment
        guards used for generated answers. Only a fully grounded/aligned result
        is eligible to skip generation.
        """
        raw_question = str(question or "")
        if not cls.looks_like_c_cpp(raw_question):
            return "", {"reason": "not_visible_code"}

        candidates = [
            item
            for item in (results or [])
            if isinstance(item, Mapping)
            and (
                item.get("_misra_rule_body_rescue") is True
                or bool(item.get("_structured_topic_family"))
            )
            and str(item.get("_misra_cue", "") or "").strip()
            and cls._is_citable_rule_body_record(item)
        ]
        if not candidates:
            return "", {"reason": "no_deterministic_candidates"}

        # Optimization #1 intentionally covered only multi-rule reviewer paths.
        # Optimization #2 adds one narrow single-rule case: an evidence-constrained
        # reviewer request that explicitly tells DocuBot not to invent/assume facts.
        # This keeps ordinary single-rule reviews on Qwen for naturalness while
        # eliminating the measured OC-008 long tail when the visible code already
        # establishes a definite source-grounded state.
        single_rule_evidence_constrained = False
        if len(candidates) == 1:
            clean_question = re.sub(r"\s+", " ", raw_question.strip().casefold())
            single_rule_evidence_constrained = bool(re.search(
                r"\b(?:do\s+not|don't|dont)\s+(?:invent|assume)\b"
                r"|\bwithout\s+(?:inventing|assuming)\b"
                r"|\b(?:huwag|wag)\s+(?:mag[- ]?)?(?:imbento|assume)\b"
                r"|\bbased\s+(?:only\s+)?on\s+(?:the\s+)?(?:visible|shown|provided)\s+(?:code|snippet|evidence)\b",
                clean_question,
                re.IGNORECASE,
            ))
            if not single_rule_evidence_constrained:
                return "", {"reason": "single_rule_review_keeps_existing_route"}
        elif len(candidates) < 1:
            return "", {"reason": "no_deterministic_candidates"}

        for item in candidates:
            state = str(item.get("_misra_assessment_state", "") or "").casefold()
            if state not in {"violation", "compliant", "uncertain"}:
                return "", {"reason": "missing_deterministic_state"}
            coverage = float(item.get("_misra_cue_coverage", 0.0) or 0.0)
            minimum_coverage = 0.95 if single_rule_evidence_constrained else 0.72
            if coverage < minimum_coverage:
                return "", {"reason": "weak_cue_coverage"}
            if single_rule_evidence_constrained and state == "uncertain":
                return "", {"reason": "single_rule_uncertain_keeps_existing_route"}
            if not str(item.get("_misra_observation", "") or "").strip():
                return "", {"reason": "missing_visible_code_observation"}

        answer = cls.deterministic_assessment(raw_question, candidates)
        if not answer:
            return "", {"reason": "deterministic_assessment_unavailable"}
        if not cls.references_are_grounded(answer, candidates):
            return "", {"reason": "deterministic_references_not_grounded"}

        aligned, details = cls.generated_assessment_is_aligned(
            raw_question,
            answer,
            candidates,
        )
        if not aligned:
            return "", {
                "reason": "deterministic_alignment_failed",
                "alignment": details,
            }

        return answer, {
            "reason": (
                "single_rule_evidence_constrained_assessment_passed_existing_grounding_and_semantic_guards"
                if single_rule_evidence_constrained
                else "deterministic_assessment_passed_existing_grounding_and_semantic_guards"
            ),
            "references": sorted(cls.available_references(candidates)),
            "alignment": details,
        }

    @classmethod
    def _answer_reference_segments(cls, answer: str, reference: str) -> list[str]:
        """Return exact per-reference answer blocks, independent of Markdown bullets.

        Older segmentation expected a Rule/Directive reference to start directly
        after a newline. Markdown list prefixes such as ``- **Rule 16.2**`` broke
        that assumption and could let uncertainty text from a later rule mask an
        earlier false satisfied claim. Parse all visible references first, then
        slice from each exact reference to the next one.
        """
        text = str(answer or "")
        wanted = re.match(
            r"(?i)^\s*(?P<kind>Rule|Directive|Dir)\s+(?P<identifier>\d+(?:\.\d+)*)\s*$",
            str(reference or ""),
        )
        if not text or not wanted:
            return []
        wanted_kind = wanted.group("kind").casefold()
        if wanted_kind == "dir":
            wanted_kind = "directive"
        wanted_id = wanted.group("identifier")

        refs = list(_REF_RE.finditer(text))
        blocks: list[str] = []
        for index, match in enumerate(refs):
            kind = str(match.group("kind") or "").casefold()
            if kind == "dir":
                kind = "directive"
            identifier = str(match.group("identifier") or "")
            if kind != wanted_kind or identifier != wanted_id:
                continue
            end = refs[index + 1].start() if index + 1 < len(refs) else len(text)
            blocks.append(text[match.start():end].strip())
        return blocks

    @classmethod
    def _visible_switch_facts(cls, question: str) -> dict:
        """Mechanically extract facts that are explicit in the current switch snippet.

        These facts are deliberately syntax-only. They do not infer hidden types or
        overall MISRA compliance; they only protect exact user-visible details such
        as the expression text, number of case/default labels, and default position.
        """
        raw = str(question or "")
        code = cls._code_only_view(raw) or raw
        match = re.search(r"\bswitch\s*\(\s*(?P<expr>[^)]*?)\s*\)\s*\{", code, re.IGNORECASE)
        if not match:
            return {}

        brace_start = match.end() - 1
        depth = 0
        body_end = -1
        quote = ""
        escape = False
        for index in range(brace_start, len(code)):
            char = code[index]
            if quote:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == quote:
                    quote = ""
                continue
            if char in {'"', "'"}:
                quote = char
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    body_end = index
                    break
        if body_end < 0:
            return {}

        body = code[brace_start + 1:body_end]
        labels = list(re.finditer(r"\b(?P<kind>case\b[^:]*|default)\s*:", body, re.IGNORECASE))
        case_count = sum(1 for label in labels if label.group("kind").lstrip().casefold().startswith("case"))
        default_indices = [
            index for index, label in enumerate(labels)
            if label.group("kind").lstrip().casefold().startswith("default")
        ]
        default_position = ""
        if default_indices:
            pos = default_indices[0]
            if len(labels) == 1:
                default_position = "only"
            elif pos == 0:
                default_position = "first"
            elif pos == len(labels) - 1:
                default_position = "last"
            else:
                default_position = "middle"

        return {
            "expression": re.sub(r"\s+", " ", match.group("expr")).strip(),
            "expression_visible": bool(match.group("expr").strip()),
            "case_label_count": case_count,
            "default_label_count": len(default_indices),
            "label_count": len(labels),
            "default_position": default_position,
        }

    @classmethod
    def normalize_generated_status_contract(
        cls,
        question: str,
        answer: str,
        results: Sequence[Mapping],
    ) -> tuple[str, list[dict]]:
        """Add a missing canonical status label without rewriting LLM prose.

        The deterministic assessment state is already established before model
        generation. Small local models sometimes explain the right relationship
        but omit the explicit polarity wording required by the semantic guard.
        This method fills only that *missing label*. It refuses to normalize a
        block that already contains an explicit contradictory polarity, so a
        real hallucination still fails the downstream guard.
        """

        text = str(answer or "")
        if not text.strip():
            return text, []

        uncertain_re = re.compile(
            r"(?i)(?:needs?\s+more\s+context|more\s+context|"
            r"cannot\s+(?:be\s+)?confirm|cannot\s+establish|"
            r"depends?\s+on|insufficient|not\s+enough|"
            r"kailangan(?:\s+pa)?\s+(?:ng\s+)?context|hindi\s+pa\s+ma[- ]?confirm)"
        )
        violation_re = re.compile(
            r"(?i)(?:non[- ]?compliant|non[- ]?compliance|violat(?:e|es|ed|ion)|"
            r"potential\s+(?:misra\s+)?(?:non[- ]?compliance|violation))"
        )
        strong_violation_re = re.compile(
            r"(?i)(?:non[- ]?compliant|non[- ]?compliance|"
            r"potential\s+(?:misra\s+)?(?:non[- ]?compliance|violation)|"
            r"\b(?:is|are|would\s+be|constitutes?)\s+"
            r"(?:(?:definitely|clearly|certainly|confirmed)\s+)?(?:a\s+)?violation\b|"
            r"\bviolates?\b)"
        )
        compliant_re = re.compile(
            r"(?i)(?:satisfied|\bcompliant\b|no\s+(?:confirmed\s+)?violation|"
            r"no\s+non[- ]?compliance|does\s+not\s+violate|walang\s+violation)"
        )
        potential_status_re = re.compile(
            r"(?i)(?:status\s*[:\-]\s*)?potential\s+(?:misra\s+)?(?:non[- ]?compliance|violation)"
        )
        uncertain_status_re = re.compile(
            r"(?i)(?:status\s*[:\-]\s*)?(?:needs?\s+more\s+context|cannot\s+(?:be\s+)?confirmed?)"
        )
        compliant_status_re = re.compile(
            r"(?i)(?:status\s*[:\-]\s*)?(?:satisfied(?:\s*/\s*no\s+violation\s+established)?|compliant|no\s+violation\s+established)"
        )

        def explicit_compliant_claim(block: str) -> bool:
            """Return True only for an affirmative compliance/satisfaction claim.

            A bare word such as ``satisfied`` inside uncertainty prose (for
            example, ``does not establish whether this requirement is satisfied``)
            is not a positive status.  Evaluate line-by-line so explicit Status or
            conclusion lines still count while uncertainty-qualified wording does not.
            """
            for raw_line in str(block or "").splitlines() or [str(block or "")]:
                line = raw_line.strip()
                if not line or not compliant_status_re.search(line):
                    continue
                uncertainty_qualified = bool(uncertain_re.search(line))
                uncertainty_qualified = uncertainty_qualified or bool(re.search(
                    r"(?i)\b(?:does\s+not\s+establish|not\s+(?:yet\s+)?established)\b",
                    line,
                ))
                uncertainty_qualified = uncertainty_qualified or bool(re.search(
                    r"(?i)\b(?:whether|if)\b.{0,120}\b(?:is|are|was|were)?\s*satisfied\b",
                    line,
                ))
                uncertainty_qualified = uncertainty_qualified or bool(re.search(
                    r"(?i)\bnot\s+satisfied\b",
                    line,
                ))
                if uncertainty_qualified:
                    continue
                return True
            return False
        confirmed_violation_re = re.compile(
            r"(?i)(?<!potential\s)(?<!potential\smisra\s)(?:\bnon[- ]?compliance\b|\bnon[- ]?compliant\b|\bviolates?\b|\bconfirmed\s+violation\b)"
        )

        def occurrence_segments(reference: str) -> list[str]:
            return cls._answer_reference_segments(text, reference)

        changes: list[dict] = []
        candidates = [
            item for item in (results or [])
            if isinstance(item, Mapping)
            and (
                item.get("_misra_rule_body_rescue") is True
                or (
                    bool(item.get("_structured_topic_family"))
                    and "_misra_assessment_state" in item
                )
            )
            and cls._is_citable_rule_body_record(item)
        ]

        for item in candidates:
            metadata = item.get("metadata", {}) or {}
            kind = (
                "Directive"
                if str(metadata.get("section_type", "")).casefold() == "directive"
                else "Rule"
            )
            identifier = str(
                metadata.get("directive_id", "")
                or metadata.get("rule_id", "")
                or ""
            ).strip()
            reference = f"{kind} {identifier}"
            expected_state = str(
                item.get("_misra_assessment_state", "violation") or "violation"
            ).casefold()

            segments = cls._answer_reference_segments(text, reference)
            if not segments:
                continue
            segment = segments[0]
            match = re.search(re.escape(reference), text, re.IGNORECASE)
            if not match:
                continue

            canonical = ""
            contradiction = False
            already_present = False
            if expected_state == "uncertain":
                canonical = "Needs more context"
                already_present = bool(uncertain_re.search(segment))
                contradiction = bool(
                    violation_re.search(segment)
                    and not uncertain_re.search(segment)
                )
            elif expected_state == "compliant":
                canonical = "Satisfied / no violation established"
                already_present = bool(compliant_re.search(segment))
                contradiction = bool(
                    strong_violation_re.search(segment)
                    and not compliant_re.search(segment)
                )
            else:
                canonical = "Potential non-compliance"
                already_present = bool(strong_violation_re.search(segment))
                contradiction = bool(
                    compliant_re.search(segment)
                    and not strong_violation_re.search(segment)
                )

            if already_present or contradiction or not canonical:
                continue

            line_end = text.find("\n", match.end())
            if line_end < 0:
                line_end = len(text)
            insertion = f"\n- **Status:** {canonical}"
            text = text[:line_end] + insertion + text[line_end:]
            changes.append({
                "reference": reference,
                "inserted_status": canonical,
            })

        return text, changes

    @classmethod
    def sanitize_generated_macro_reconstruction(
        cls,
        question: str,
        answer: str,
    ) -> tuple[str, list[dict]]:
        """Remove model-invented corrected macro expansions from explanations.

        For a visibly unsafe function-like macro, literal substitution is
        mechanically known.  If the user did not request a fix, a model must
        not introduce a different parenthesized expansion and present it as the
        intended result.  The sanitizer removes only the sentence containing
        that derived corrected form; the literal expansion and source-grounded
        explanation remain untouched.
        """

        text = str(answer or "")
        raw_question = str(question or "")
        if not text.strip():
            return text, []

        remediation_requested = bool(re.search(
            r"(?i)\b(?:fix|correct|rewrite|remed(?:y|iate)|solution|"
            r"aayusin|ayusin|baguhin|compliant\s+(?:version|rewrite|form)|"
            r"how\s+(?:should|can)\s+i\s+(?:fix|rewrite))\b",
            raw_question,
        ))
        if remediation_requested:
            return text, []

        changes: list[dict] = []
        for macro in cls._simple_function_macro_expansions(raw_question):
            unsafe_params = set(macro.get("unsafe_params") or [])
            if not unsafe_params:
                continue

            body = str(macro.get("body") or "")
            params = list(macro.get("params") or [])
            invocation = str(macro.get("invocation") or "")
            if not body or not params or not invocation:
                continue

            open_index = invocation.find("(")
            close_index = invocation.rfind(")")
            if open_index < 0 or close_index <= open_index:
                continue
            args = cls._split_top_level_arguments(
                invocation[open_index + 1:close_index]
            )
            if len(args) != len(params):
                continue

            corrected = body
            for param, argument in zip(params, args):
                replacement = (
                    f"({argument})"
                    if param in unsafe_params
                    else argument
                )
                corrected = re.sub(
                    rf"(?<![A-Za-z0-9_]){re.escape(param)}(?![A-Za-z0-9_])",
                    replacement,
                    corrected,
                )
            corrected = re.sub(r"\s+", " ", corrected).strip()
            literal = re.sub(r"\s+", " ", str(macro.get("expanded") or "")).strip()
            if not corrected or corrected == literal:
                continue

            compact_candidate = re.sub(r"\s+", "", corrected.replace("`", ""))
            if not compact_candidate:
                continue

            kept_parts: list[str] = []
            removed = False
            for line in text.splitlines():
                pieces = re.split(r"(?<=[.!?])\s+", line)
                line_parts: list[str] = []
                for piece in pieces:
                    compact_piece = re.sub(r"\s+", "", piece.replace("`", ""))
                    if compact_candidate in compact_piece:
                        removed = True
                        continue
                    line_parts.append(piece)
                kept_parts.append(" ".join(part for part in line_parts if part).strip())

            if removed:
                text = "\n".join(kept_parts)
                changes.append({
                    "macro": str(macro.get("name") or ""),
                    "removed_reconstruction": corrected,
                })

        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text, changes

    @classmethod
    def generated_assessment_is_aligned(
        cls,
        question: str,
        answer: str,
        results: Sequence[Mapping],
    ) -> tuple[bool, dict]:
        """Validate LLM MISRA status/relationship against deterministic evidence.

        This guard is deliberately semantic rather than citation-only.  It checks
        each citable Rule/Directive against the deterministic visible-code state
        derived before generation, so a model cannot turn ``Needs more context``
        into a confirmed violation merely while citing the correct Rule number.
        """

        text = str(answer or "")
        if not text.strip():
            return False, {"reason": "empty_answer"}

        failures = []
        candidates = [
            item for item in (results or [])
            if isinstance(item, Mapping)
            and (
                item.get("_misra_rule_body_rescue") is True
                or (
                    bool(item.get("_structured_topic_family"))
                    and "_misra_assessment_state" in item
                )
            )
            and cls._is_citable_rule_body_record(item)
        ]
        if not candidates:
            # Generic retrieved MISRA chunks still rely on the existing strict
            # reference-grounding guard. No deterministic code-state exists here.
            return True, {"checked": 0, "failures": []}

        def segment_for(reference: str) -> str:
            blocks = cls._answer_reference_segments(text, reference)
            return blocks[0] if blocks else ""

        uncertain_re = re.compile(
            r"(?i)(?:needs?\s+more\s+context|more\s+context|"
            r"cannot\s+(?:be\s+)?confirm|cannot\s+establish|"
            r"depends?\s+on|insufficient|not\s+enough|"
            r"kailangan(?:\s+pa)?\s+(?:ng\s+)?context|hindi\s+pa\s+ma[- ]?confirm)"
        )
        violation_re = re.compile(
            r"(?i)(?:non[- ]?compliant|non[- ]?compliance|violat(?:e|es|ed|ion)|"
            r"potential\s+(?:misra\s+)?(?:non[- ]?compliance|violation))"
        )
        strong_violation_re = re.compile(
            r"(?i)(?:non[- ]?compliant|non[- ]?compliance|"
            r"potential\s+(?:misra\s+)?(?:non[- ]?compliance|violation)|"
            r"\b(?:is|are|would\s+be|constitutes?)\s+(?:a\s+)?violation\b|"
            r"\bviolates?\b)"
        )
        compliant_re = re.compile(
            r"(?i)(?:satisfied|\bcompliant\b|no\s+(?:confirmed\s+)?violation|"
            r"no\s+non[- ]?compliance|does\s+not\s+violate|walang\s+violation)"
        )
        potential_status_re = re.compile(
            r"(?i)(?:status\s*[:\-]\s*)?potential\s+(?:misra\s+)?(?:non[- ]?compliance|violation)"
        )
        uncertain_status_re = re.compile(
            r"(?i)(?:status\s*[:\-]\s*)?(?:needs?\s+more\s+context|cannot\s+(?:be\s+)?confirmed?)"
        )
        compliant_status_re = re.compile(
            r"(?i)(?:status\s*[:\-]\s*)?(?:satisfied(?:\s*/\s*no\s+violation\s+established)?|compliant|no\s+violation\s+established)"
        )

        def explicit_compliant_claim(block: str) -> bool:
            """Return True only for an affirmative compliance/satisfaction claim.

            A bare word such as ``satisfied`` inside uncertainty prose (for
            example, ``does not establish whether this requirement is satisfied``)
            is not a positive status.  Evaluate line-by-line so explicit Status or
            conclusion lines still count while uncertainty-qualified wording does not.
            """
            for raw_line in str(block or "").splitlines() or [str(block or "")]:
                line = raw_line.strip()
                if not line or not compliant_status_re.search(line):
                    continue
                uncertainty_qualified = bool(uncertain_re.search(line))
                uncertainty_qualified = uncertainty_qualified or bool(re.search(
                    r"(?i)\b(?:does\s+not\s+establish|not\s+(?:yet\s+)?established)\b",
                    line,
                ))
                uncertainty_qualified = uncertainty_qualified or bool(re.search(
                    r"(?i)\b(?:whether|if)\b.{0,120}\b(?:is|are|was|were)?\s*satisfied\b",
                    line,
                ))
                uncertainty_qualified = uncertainty_qualified or bool(re.search(
                    r"(?i)\bnot\s+satisfied\b",
                    line,
                ))
                if uncertainty_qualified:
                    continue
                return True
            return False
        confirmed_violation_re = re.compile(
            r"(?i)(?<!potential\s)(?<!potential\smisra\s)(?:\bnon[- ]?compliance\b|\bnon[- ]?compliant\b|\bviolates?\b|\bconfirmed\s+violation\b)"
        )
        explicit_confirmed_status_re = re.compile(
            r"(?i)(?:\bstatus\b\s*[:\-]?\s*(?:\*+\s*)?(?:confirmed\s+)?(?:non[- ]?compliance|non[- ]?compliant|violation)\b|"
            r"^\s*#{1,6}\s*(?:\*+\s*)?(?:confirmed\s+)?(?:non[- ]?compliance|non[- ]?compliant|violation)\s*(?:\*+)?\s*$)",
            re.MULTILINE,
        )

        def occurrence_segments(reference: str) -> list[str]:
            return cls._answer_reference_segments(text, reference)

        for item in candidates:
            metadata = item.get("metadata", {}) or {}
            kind = "Directive" if str(metadata.get("section_type", "")).casefold() == "directive" else "Rule"
            identifier = str(metadata.get("directive_id", "") or metadata.get("rule_id", "") or "").strip()
            reference = f"{kind} {identifier}"
            expected_state = str(item.get("_misra_assessment_state", "violation") or "violation").casefold()
            segment = segment_for(reference)
            if not segment:
                failures.append({"reference": reference, "reason": "missing_reference_segment", "expected_state": expected_state})
                continue

            blocks = occurrence_segments(reference) or [segment]

            if expected_state == "uncertain":
                if not uncertain_re.search(segment):
                    failures.append({"reference": reference, "reason": "missing_uncertainty_language", "expected_state": expected_state})
                for block in blocks:
                    if potential_status_re.search(block) or confirmed_violation_re.search(block):
                        failures.append({"reference": reference, "reason": "uncertain_explicit_status_promoted", "expected_state": expected_state})
                        failures.append({"reference": reference, "reason": "uncertain_promoted_to_violation", "expected_state": expected_state})
                        break
                    if explicit_compliant_claim(block):
                        failures.append({"reference": reference, "reason": "uncertain_promoted_to_compliant", "expected_state": expected_state})
                        break

            elif expected_state == "compliant":
                if not compliant_re.search(segment):
                    failures.append({"reference": reference, "reason": "missing_compliant_state", "expected_state": expected_state})
                for block in blocks:
                    if potential_status_re.search(block) or confirmed_violation_re.search(block):
                        failures.append({"reference": reference, "reason": "compliant_explicit_status_promoted", "expected_state": expected_state})
                        break

            else:
                # The deterministic ``violation`` state is intentionally exposed
                # to users as *Potential non-compliance*.  Reject both a downgrade
                # to uncertainty/compliance and an escalation to confirmed
                # non-compliance. This catches contradictory duplicated status
                # headings as well as overconfident conclusions.
                if not potential_status_re.search(segment):
                    failures.append({"reference": reference, "reason": "missing_potential_noncompliance_state", "expected_state": expected_state})
                if compliant_re.search(segment) and not potential_status_re.search(segment):
                    failures.append({"reference": reference, "reason": "violation_promoted_to_compliant", "expected_state": expected_state})
                for block in blocks:
                    if uncertain_status_re.search(block):
                        failures.append({"reference": reference, "reason": "potential_status_downgraded_to_uncertain", "expected_state": expected_state})
                        break
                    if compliant_status_re.search(block) and not potential_status_re.search(block):
                        failures.append({"reference": reference, "reason": "potential_status_promoted_to_compliant", "expected_state": expected_state})
                        break
                    if explicit_confirmed_status_re.search(block):
                        failures.append({"reference": reference, "reason": "potential_status_escalated_to_confirmed_violation", "expected_state": expected_state})
                        break
                    if confirmed_violation_re.search(block) and not potential_status_re.search(block):
                        failures.append({"reference": reference, "reason": "potential_status_escalated_to_confirmed_violation", "expected_state": expected_state})
                        break

        # A conceptual/text-only question must never acquire a fabricated
        # "user code" snippet from a source example or model memory.  Reject
        # explicit claims that code was provided when the current user turn
        # contains no C/C++ syntax.
        if not cls.looks_like_c_cpp(question):
            invented_user_code = bool(
                re.search(
                    r"(?i)\b(?:the\s+)?user(?:'s)?\s+(?:code|snippet)|"
                    r"\bprovided\s+(?:code|snippet)|\bvisible\s+(?:code|snippet|construct)|"
                    r"\bspecific\s+construct\s+is\b",
                    text,
                )
                and ("```" in text or re.search(r"[;{}]|&&|\|\|", text))
            )
            if invented_user_code:
                failures.append({
                    "reference": "user input",
                    "reason": "invented_user_code_for_text_only_question",
                })

        # Never let a model promote a rule-scoped satisfied result into a blanket
        # claim that the entire visible construct is "MISRA-compliant".  The
        # retrieved evidence may prove only one or a few specific requirements;
        # a whole-construct compliance claim would therefore exceed the evidence.
        # Scoped wording such as "complies with Rule 20.7" remains allowed.
        if re.search(r"(?i)\bMISRA[- ]compliant\b", text):
            failures.append({
                "reference": "overall assessment",
                "reason": "blanket_misra_compliance_overclaim",
            })

        # Preserve the exact visible operator for mechanically observed pointer
        # arithmetic. A model must not turn ``p = p + 2`` into ``p += 2`` (or the
        # reverse) while still citing the correct Rule.  The requirement may list
        # all covered operators; only an explicit claim about the visible/current
        # construct is compared here.
        for item in candidates:
            expected_observation = str(item.get("_misra_observation", "") or "")
            expected_match = re.search(
                r"(?i)\buses\s+the\s+`(?P<operator>\+=|-=|\+|-)`\s+operator\b",
                expected_observation,
            )
            if not expected_match:
                continue
            metadata = item.get("metadata", {}) or {}
            kind = "Directive" if str(metadata.get("section_type", "")).casefold() == "directive" else "Rule"
            identifier = str(metadata.get("directive_id", "") or metadata.get("rule_id", "") or "").strip()
            reference = f"{kind} {identifier}".strip()
            segment = segment_for(reference) if reference else text
            visible_claim = re.search(
                r"(?is)(?:visible|current|shown|this)\b.{0,180}?"
                r"(?:appl(?:y|ies)|uses?)\s+(?:the\s+)?`?(?P<operator>\+=|-=|\+|-)`?"
                r"(?:\s+operator|\s+to\b)",
                segment,
            )
            if visible_claim and visible_claim.group("operator") != expected_match.group("operator"):
                failures.append({
                    "reference": reference or "visible operator",
                    "reason": "visible_operator_mismatch",
                    "expected_operator": expected_match.group("operator"),
                    "claimed_operator": visible_claim.group("operator"),
                })

        # Source rationale must not be presented as a visible event when the
        # user's code does not contain that construct.  This catches the prior
        # Rule 15.6 response that claimed a semicolon after the control expression.
        code = cls._code_only_view(str(question or ""))
        if re.search(r"(?i)semi[- ]?colon\s+after\s+the\s+controlling\s+expression", text):
            if not re.search(r"\b(?:if|for|while)\s*\([^)]*\)\s*;", code, re.IGNORECASE):
                failures.append({"reference": "Rule 15.6", "reason": "source_rationale_misstated_as_visible_code"})

        # When the model explicitly states a visible function-like macro
        # expansion, verify it against mechanical substitution of the exact
        # shown macro body/arguments. This blocks inserted parentheses or other
        # code transformations that are not present in the user's snippet.
        for macro in cls._simple_function_macro_expansions(str(question or "")):
            if not macro.get("unsafe_params"):
                continue
            expansion_claim = re.search(
                r"(?is)\bexpand(?:s|ed|ing)?\s+to\b(?P<tail>.{0,240})",
                text,
            )
            if not expansion_claim:
                continue
            tail = str(expansion_claim.group("tail") or "")
            # Only police code-like claims, not a purely verbal explanation.
            if not re.search(r"[+*/%<>&|^-]", tail):
                continue
            expected = re.sub(
                r"\s+",
                "",
                str(macro.get("expanded") or "").replace("`", ""),
            )
            claimed_window = re.sub(
                r"\s+",
                "",
                tail.replace("`", ""),
            )
            if expected and expected not in claimed_window:
                failures.append({
                    "reference": "macro expansion",
                    "reason": "macro_expansion_claim_mismatch",
                    "expected_expansion": str(macro.get("expanded") or ""),
                    "invocation": str(macro.get("invocation") or ""),
                })
                break

        _sanitized_macro_text, nonliteral_reconstructions = (
            cls.sanitize_generated_macro_reconstruction(
                question,
                text,
            )
        )
        for reconstruction in nonliteral_reconstructions:
            failures.append({
                "reference": "macro expansion",
                "reason": "invented_nonliteral_macro_reconstruction",
                **reconstruction,
            })

        # Mechanically visible switch facts must survive generation verbatim.
        # This protects label counts/positions and prevents wording that claims
        # the visible switch expression itself is absent when only its type is unknown.
        switch_facts = cls._visible_switch_facts(question)
        if switch_facts:
            case_count = int(switch_facts.get("case_label_count", 0) or 0)
            default_position = str(switch_facts.get("default_position", "") or "").casefold()
            expression = str(switch_facts.get("expression", "") or "").strip()

            number_words = {
                "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10,
            }
            for count_claim in re.finditer(
                r"(?i)\b(?:has|have|contains?|includes?|with)\s+(?:at\s+least\s+)?"
                r"(?P<count>\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
                r"`?case`?\s+labels?\b",
                text,
            ):
                raw_count = count_claim.group("count").casefold()
                claimed = int(raw_count) if raw_count.isdigit() else number_words.get(raw_count)
                if claimed is not None and claimed != case_count:
                    failures.append({
                        "reference": "visible switch facts",
                        "reason": "case_label_count_mismatch",
                        "expected": case_count,
                        "claimed": claimed,
                    })
                    break

            if default_position in {"first", "last", "middle"}:
                opposite_positions = {
                    "first": ("last", "middle"),
                    "last": ("first", "middle"),
                    "middle": ("first", "last"),
                }[default_position]
                default_claim = re.compile(
                    r"(?i)\bdefault(?:\s+label)?\b.{0,45}\b(first|last|middle)\b|"
                    r"\b(first|last|middle)\b.{0,45}\bdefault(?:\s+label)?\b"
                )
                for position_claim in default_claim.finditer(text):
                    claimed = (position_claim.group(1) or position_claim.group(2) or "").casefold()
                    if claimed in opposite_positions:
                        failures.append({
                            "reference": "visible switch facts",
                            "reason": "default_position_mismatch",
                            "expected": default_position,
                            "claimed": claimed,
                        })
                        break

            if expression and re.search(
                r"(?i)\bswitch[- ]expression\b.{0,45}\b(?:not\s+(?:explicitly\s+)?visible|"
                r"isn['’]?t\s+(?:explicitly\s+)?visible|not\s+shown|not\s+provided)\b",
                text,
            ):
                failures.append({
                    "reference": "visible switch facts",
                    "reason": "visible_switch_expression_denied",
                    "expression": expression,
                })

        # If the model volunteers a corrected definition for a visibly unsafe
        # function-like macro, verify that the suggested replacement actually
        # encloses each affected parameter occurrence.  This is a safety backup
        # for the generation contract; it does not synthesize a correction.
        for macro in cls._simple_function_macro_expansions(str(question or "")):
            unsafe_params = list(macro.get("unsafe_params") or [])
            name = str(macro.get("name") or "").strip()
            if not unsafe_params or not name:
                continue
            suggested = re.search(
                rf"(?m)^\s*#\s*define\s+{re.escape(name)}\s*"
                r"\((?P<params>[^)]*)\)\s+(?P<body>[^\n]+)",
                text,
            )
            if not suggested:
                continue
            body = str(suggested.group("body") or "").strip().strip("`")
            bad_params = []
            for param in unsafe_params:
                occurrences = list(
                    re.finditer(
                        rf"(?<![A-Za-z0-9_]){re.escape(param)}(?![A-Za-z0-9_])",
                        body,
                    )
                )
                if any(
                    not (
                        body[:occ.start()].rstrip().endswith("(")
                        and body[occ.end():].lstrip().startswith(")")
                    )
                    for occ in occurrences
                ):
                    bad_params.append(param)
            if bad_params:
                failures.append({
                    "reference": "macro remediation",
                    "reason": "suggested_macro_fix_leaves_parameter_expansion_unparenthesized",
                    "macro": name,
                    "parameters": bad_params,
                })
                break

        return not failures, {"checked": len(candidates), "failures": failures}

    @classmethod
    def annotate_structured_family_assessment(
        cls,
        question: str,
        results: Sequence[Mapping],
    ) -> list[Mapping]:
        """Attach conservative deterministic states to structured-family evidence.

        Complete Rule-family retrieval intentionally bypasses the bounded semantic
        rescue path.  That means those records may arrive without the deterministic
        state fields used by the generation contract.  Derive them from the current
        user turn and the authoritative Rule statement before any LLM synthesis.

        When the existing interpreter has no construct-specific proof and merely
        echoes the Rule statement as its historical fallback observation, downgrade
        that item to ``uncertain`` rather than treating absence of proof as a
        violation.  This prevents a model from declaring unknown facts satisfied or
        violated while preserving specific decidable checks such as Rules 16.4-16.7.
        """
        enriched_results: list[Mapping] = []
        for item in results or []:
            if not isinstance(item, Mapping):
                enriched_results.append(item)
                continue
            enriched = dict(item)
            enriched["metadata"] = dict(item.get("metadata", {}) or {})
            if not cls._is_citable_rule_body_record(enriched):
                enriched_results.append(enriched)
                continue
            statement = cls._source_rule_statement(str(enriched.get("text", "") or ""))
            if not statement:
                enriched_results.append(enriched)
                continue
            state, observation = cls.cue_assessment_state(question, statement)
            if (
                state == "violation"
                and cls._normalized_phrase(observation) == cls._normalized_phrase(statement)
            ):
                state = "uncertain"
                observation = (
                    "The matched requirement is relevant, but the visible information "
                    "does not establish whether this requirement is satisfied."
                )
            enriched["_misra_cue"] = statement
            enriched["_misra_cue_coverage"] = 1.0
            enriched["_misra_assessment_state"] = state
            enriched["_misra_observation"] = observation
            enriched_results.append(enriched)
        return enriched_results

    @classmethod
    def build_generation_contract(
        cls,
        question: str,
        results: Sequence[Mapping],
    ) -> str:
        """Build a compact evidence-derived contract for grounded LLM synthesis.

        The contract never invents Rule/Directive identifiers.  It only repeats
        citable identifiers and deterministic visible-code states already attached
        to accepted MISRA rule-body evidence.  Its purpose is to let the LLM
        explain the evidence naturally without changing polarity, borrowing code
        from chat history, or fabricating remediation.
        """

        rows: list[str] = []
        for item in results or []:
            if not isinstance(item, Mapping):
                continue
            if not (
                item.get("_misra_rule_body_rescue") is True
                or (
                    bool(item.get("_structured_topic_family"))
                    and "_misra_assessment_state" in item
                )
            ):
                continue
            if not cls._is_citable_rule_body_record(item):
                continue
            metadata = item.get("metadata", {}) or {}
            section_type = str(metadata.get("section_type", "") or "").casefold()
            kind = "Directive" if section_type == "directive" else "Rule"
            identifier = str(
                metadata.get("directive_id", "")
                or metadata.get("rule_id", "")
                or ""
            ).strip()
            if not identifier:
                continue
            state = str(item.get("_misra_assessment_state", "violation") or "violation").casefold()
            state_label = {
                "violation": "POTENTIAL NON-COMPLIANCE",
                "uncertain": "NEEDS MORE CONTEXT",
                "compliant": "SATISFIED / NO VIOLATION ESTABLISHED",
            }.get(state, "POTENTIAL NON-COMPLIANCE")
            observation = re.sub(
                r"\s+",
                " ",
                str(item.get("_misra_observation", "") or ""),
            ).strip()
            row = f"- {kind} {identifier} | REQUIRED STATUS: {state_label}"
            if observation:
                row += f" | VISIBLE FACT: {observation}"
            rows.append(row)

        if not rows:
            return ""

        extra: list[str] = []
        macro_expansions = cls._simple_function_macro_expansions(question)
        for macro in macro_expansions:
            if not macro.get("unsafe_params"):
                continue
            extra.append(
                "- LITERAL MACRO EXPANSION: "
                f"{macro.get('invocation', '')} -> {macro.get('expanded', '')}. "
                "If the user did not ask for a fix, do not invent corrected code. "
                "If a fix is requested, do not claim that wrapping only the whole "
                "replacement list fixes parameter-expansion parentheses; each affected "
                "parameter occurrence must remain parenthesized in any suggested rewrite."
            )

        switch_facts = cls._visible_switch_facts(question)
        if switch_facts:
            expression = str(switch_facts.get("expression", "") or "").strip()
            extra.append(
                "- CURRENT SWITCH FACTS (syntax-only; do not alter): "
                f"visible expression=`{expression}`; "
                f"case labels={switch_facts.get('case_label_count', 0)}; "
                f"default labels={switch_facts.get('default_label_count', 0)}; "
                f"total visible labels={switch_facts.get('label_count', 0)}; "
                f"default position={switch_facts.get('default_position', '') or 'unknown'}. "
                "Expression visibility does not establish its essential type."
            )

        instructions = [
            "MISRA EVIDENCE-DERIVED GENERATION CONTRACT:",
            *rows,
            *extra,
            "Contract rules:",
            "1. Preserve every REQUIRED STATUS exactly for the matching Rule/Directive.",
            "2. Do not use a blanket overall violation statement that makes a NEEDS MORE CONTEXT item sound confirmed.",
            "3. In the conclusion, repeat each Rule/Directive with the same polarity as its REQUIRED STATUS.",
            "4. Explain only the current USER CODE / SCENARIO. Do not borrow code, functions, variables, or hazards from earlier chat turns unless the current turn is an explicit grounded follow-up.",
            "5. Do not invent corrected code, casts, parentheses, values, side effects, or consequences unless the user asks for remediation and the accepted evidence supports it.",
            "6. For each Rule/Directive, output exactly one status polarity. Never add a second Status heading or later change POTENTIAL NON-COMPLIANCE to Non-Compliance/Violation, NEEDS MORE CONTEXT to Potential Non-Compliance, or SATISFIED to a violation.",
            "7. A conclusion must reuse the exact canonical REQUIRED STATUS wording; do not paraphrase it into a stronger or weaker compliance claim.",
            "8. Never call the whole code, program, statement, function, macro, switch, or construct MISRA-compliant. A satisfied result is scoped only to the identified Rule/Directive evidence; say which Rule/Directive is satisfied or that no violation is established for that requirement.",
            "9. Preserve CURRENT SWITCH FACTS exactly. Do not confuse case-label count with switch-clause count, do not change first/last/default position, and never say a visible switch expression is absent merely because its type is unknown.",
        ]
        return "\n".join(instructions)

    @classmethod
    def should_isolate_generation_history(
        cls,
        question: str,
        *,
        grounded_followup: bool = False,
    ) -> bool:
        """Return True for a fresh self-contained MISRA assessment turn.

        User messages from prior turns are useful for true deictic follow-ups but are
        unsafe as generation evidence for a new code/scenario review.  Retrieval and
        grounded state already carry the authoritative source context, so a fresh
        MISRA question should not expose old user code to the LLM.
        """

        if grounded_followup:
            return False
        raw = str(question or "").strip()
        if not raw:
            return False
        if cls.looks_like_c_cpp(raw):
            return True
        # Text-only MISRA scenarios can still be fully self-contained.  A current
        # semantic cue means the turn can be answered from its own wording plus
        # retrieved company evidence, without borrowing previous code.
        return bool(cls.semantic_cues(raw))

    @staticmethod
    def _context_text_for_intent(text: str, question: str, citable: str) -> str:
        """Keep only evidence roles that help the current natural question.

        Rule-body rescue can attach Rationale, Amplification, Exceptions and
        Examples to one result. Feeding every child block to generation caused
        irrelevant example/code spill into otherwise simple answers. Keep the
        authoritative body always, then admit supporting blocks only when the
        user's intent needs them.
        """

        raw = str(text or "").strip()
        clean_q = re.sub(r"\s+", " ", str(question or "").casefold()).strip()
        if not raw or not clean_q or citable == "NONE":
            return raw

        blocks = [block.strip() for block in re.split(r"\n\s*\n+", raw) if block.strip()]
        if len(blocks) <= 1:
            return raw

        asks_example = bool(re.search(r"\b(?:example|examples|sample|illustrat)\w*\b", clean_q))
        asks_reason = bool(re.search(r"\b(?:why|reason|rationale)\b", clean_q))
        asks_scope = bool(re.search(r"\b(?:amplification|scope|exception|exceptions|apply|applies|applicable)\b", clean_q))
        rich_review = bool(re.search(
            r"\b(?:review|analy[sz]e|assess|evaluate|explain|guidance|consider|check|issues?|concerns?)\b",
            clean_q,
        ))

        allowed_labels = set()
        if asks_example:
            allowed_labels.update({"example", "examples"})
        if asks_reason:
            allowed_labels.update({"rationale", "amplification"})
        if asks_scope:
            allowed_labels.update({"amplification", "exception", "exceptions", "note", "notes"})
        if rich_review and not asks_example:
            allowed_labels.update({"amplification", "rationale", "exception", "exceptions", "note", "notes"})

        kept = [blocks[0]]
        for block in blocks[1:]:
            label_match = re.match(
                r"^\s*(Amplification|Rationale|Exceptions?|Examples?|Notes?|See\s+also)\b",
                block,
                re.IGNORECASE,
            )
            if not label_match:
                continue
            label = re.sub(r"\s+", " ", label_match.group(1).casefold()).strip()
            if label in allowed_labels:
                kept.append(block)

        return "\n\n".join(kept).strip()

    @classmethod
    def build_grounded_context(
        cls,
        results: Sequence[Mapping],
        question: str = "",
    ) -> str:
        """Render MISRA context with an explicit citable-reference boundary.

        Generic sections, appendices, tables, rationale blocks, and examples
        can still support an explanation, but Rule/Directive identifiers found
        inside those blocks are not promoted to citable rule-body evidence.
        """

        blocks: list[str] = []
        for index, item in enumerate(results or [], start=1):
            metadata = item.get("metadata", {}) if isinstance(item, Mapping) else {}
            section_type = str(metadata.get("section_type", "") or "").casefold()
            rule_id = str(metadata.get("rule_id", "") or "").strip()
            directive_id = str(metadata.get("directive_id", "") or "").strip()
            section_title = str(metadata.get("section_title", "") or "").strip()
            file_name = str(metadata.get("file_name", "") or "Unknown")
            page_start = metadata.get("page_start")
            page_end = metadata.get("page_end") or page_start

            citable = "NONE"
            if section_type == "rule" and rule_id:
                citable = f"Rule {rule_id}"
            elif section_type == "directive" and (directive_id or rule_id):
                citable = f"Directive {directive_id or rule_id}"

            location = ""
            if page_start:
                location = (
                    f" | Page {page_start}"
                    if not page_end or str(page_end) == str(page_start)
                    else f" | Pages {page_start}-{page_end}"
                )

            applicability_cue = str(item.get("_misra_cue", "") or "").strip()
            cue_line = (
                f"APPLICABILITY CUE: {applicability_cue}"
                if applicability_cue
                else "APPLICABILITY CUE: NONE"
            )
            assessment_state = str(item.get("_misra_assessment_state", "") or "").strip().casefold()
            observation = re.sub(
                r"\s+", " ", str(item.get("_misra_observation", "") or "")
            ).strip()
            state_label = {
                "violation": "POTENTIAL NON-COMPLIANCE",
                "uncertain": "NEEDS MORE CONTEXT",
                "compliant": "SATISFIED / NO VIOLATION ESTABLISHED",
            }.get(assessment_state, "")
            assessment_lines = []
            if state_label:
                assessment_lines.append(
                    f"VISIBLE-CODE ASSESSMENT: {state_label}"
                )
            if observation:
                assessment_lines.append(
                    f"VISIBLE-CODE OBSERVATION: {observation}"
                )
            blocks.append(
                "\n".join(
                    [
                        f"===== MISRA EVIDENCE {index} =====",
                        f"SOURCE: {file_name}{location}",
                        f"SECTION: {section_title or 'Unlabeled'}",
                        f"CITABLE RULE/DIRECTIVE: {citable}",
                        cue_line,
                        *assessment_lines,
                        (
                            "NOTE: Identifiers mentioned inside this text are "
                            "cross-references only unless the CITABLE field above "
                            "names that exact Rule/Directive. VISIBLE-CODE fields are "
                            "deterministic observations from the user's shown code; "
                            "they are not additional source text and must not be contradicted."
                        ),
                        cls._context_text_for_intent(
                            str(item.get("text", "") or ""),
                            question,
                            citable,
                        ),
                    ]
                )
            )

        return "\n\n".join(blocks).strip()

    @staticmethod
    def _source_rule_statement(text: str) -> str:
        """Extract a Rule/Directive requirement from multiline or compact chunks.

        Prepared Option-C chunks can store the heading, requirement and labels on
        one physical line (``Rule 16.4 ... Category Required Analysis ...``).
        Treating the whole first line as a heading used to discard the requirement
        in that representation.  Parse the heading prefix separately and stop at
        source labels whether they are line-delimited or inline.
        """
        lines = [line.strip() for line in str(text or "").splitlines()]
        if not lines:
            return ""

        heading = re.match(
            r"^(?:Rule|Directive|Dir)\s+\d+(?:\.\d+)*\b\s*(?P<rest>.*)$",
            lines[0],
            re.IGNORECASE,
        )
        if heading:
            rest = str(heading.group("rest") or "").strip()
            lines = ([rest] if rest else []) + lines[1:]

        statement = []
        stop_labels = {
            "category", "analysis", "applies to", "amplification", "rationale",
            "example", "examples", "exception", "exceptions", "see also", "note", "notes",
        }
        inline_boundary = re.compile(
            r"\s+(?=(?:Category|Analysis|Applies\s+to|Amplification|Rationale|"
            r"Examples?|Exceptions?|See\s+also|Notes?)\b)",
            re.IGNORECASE,
        )
        for line in lines:
            clean = re.sub(r"\s+", " ", line).strip()
            if not clean:
                continue
            if clean.casefold() in stop_labels:
                break
            if re.match(
                r"^(?:Category|Analysis|Applies\s+to|Amplification|Rationale|"
                r"Examples?|Exceptions?|See\s+also|Notes?)\b",
                clean,
                re.IGNORECASE,
            ):
                break

            # Stop before an inline source label in compact prepared chunks.
            clean = inline_boundary.split(clean, maxsplit=1)[0].strip()
            if not clean:
                break

            # MISRA rule bodies may contain standards/reference annotations
            # between the requirement and Category. They are not part of the
            # requirement sentence/title shown to the user.
            if re.match(r"^(?:C\d{2}\b|\[[A-Za-z])", clean):
                break
            statement.append(clean)

        return re.sub(r"\s+", " ", " ".join(statement)).strip()

    @staticmethod
    def _source_labeled_block(text: str, label: str) -> str:
        """Extract a named source block from line or compact prepared chunks."""
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        if not raw.strip() or not label:
            return ""

        # Prepared structure-aware chunks frequently store a child section as
        # ``Rationale text...`` or ``Example code...`` on the same physical
        # line.  Because expanded Rule evidence joins child chunks with blank
        # lines, parse those blocks first and preserve only the requested role.
        boundary = (
            r"Category|Analysis|Applies\s+to|Amplification|Rationale|"
            r"Example|Examples|Exception|Exceptions|See\s+also|Notes?"
        )
        for block in [part.strip() for part in re.split(r"\n\s*\n+", raw) if part.strip()]:
            inline = re.match(
                rf"(?is)^\s*{re.escape(label)}\s*:?\s+(?P<body>.+)$",
                block,
            )
            if inline:
                body = str(inline.group("body") or "").strip()
                # Compact prepared chunks can append a new source role on the
                # same physical block (for example ``... See also Rule 16.1``).
                # Stop before that role so examples/rationales cannot leak
                # cross-reference identifiers into the visible answer.
                body = re.split(
                    rf"(?is)\s+(?=(?:{boundary})\s*:?\s+(?:Rule|Directive|Dir|[A-Z0-9#]))",
                    body,
                    maxsplit=1,
                )[0].strip()
                # v4 parent anchoring can leave one isolated Rule/Directive
                # header at the end of an expanded child block.  It is useful
                # retrieval metadata, but must not leak into user-facing
                # Example/Rationale text.  Strip only a terminal standalone
                # reference; references inside the substantive block remain.
                body = re.sub(
                    r"(?is)(?:\n|\s{2,})(?:Rule|Directive|Dir)\s+\d+(?:\.\d+)*\s*$",
                    "",
                    body,
                ).strip()
                return re.sub(r"[ \t]+", " ", body).strip()

        match = re.search(
            rf"(?ims)^\s*{re.escape(label)}\s*:?\s*$\s*(.+?)(?=^\s*(?:{boundary})\s*:?\s*$|\Z)",
            raw,
        )
        if not match:
            return ""
        value = str(match.group(1) or "").strip()
        # v4 expanded child chunks may end with a repeated parent anchor such
        # as ``Rule 16.5`` immediately before the next labeled block.  Remove
        # that terminal metadata-only line without touching substantive source
        # references inside the section.
        value = re.sub(
            r"(?im)\n\s*(?:Rule|Directive|Dir)\s+\d+(?:\.\d+)*\s*$",
            "",
            value,
        ).strip()
        return re.sub(r"[ \t]+", " ", value).strip()

    @staticmethod
    def _first_sentences(text: str, maximum: int = 2, max_chars: int = 700) -> str:
        clean = re.sub(r"\s+", " ", str(text or "")).strip()
        if not clean:
            return ""

        # Source rationale blocks can begin with numbered paragraphs such as
        # ``1. The use of ...``.  Treat the number as a list marker rather than
        # as a complete sentence so the UI never renders ``Rationale: 1.``.
        clean = re.sub(r"^(?:\(?\d+\)?[.)]|[-•])\s*", "", clean).strip()
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", clean)
            if part.strip()
        ]
        meaningful = [
            part for part in sentences
            if not re.fullmatch(r"(?:\(?\d+\)?[.)]?|[-•])", part)
            and len(re.sub(r"[^A-Za-z0-9]+", "", part)) >= 6
        ]
        value = " ".join(meaningful[:maximum]).strip()
        if not value:
            return ""
        if len(value) <= max_chars:
            return value
        return value[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."

    @classmethod
    def _readable_why_it_applies(
        cls,
        question: str,
        cue: str,
        observation: str,
    ) -> str:
        """Return a user-facing code/scenario relationship without inventing facts.

        Precise observations from the interpreter win.  When the interpreter has
        only a source cue, use a compact visible-code relationship rather than
        repeating the Rule statement verbatim.
        """

        observation = re.sub(r"\s+", " ", str(observation or "")).strip()
        cue_clean = re.sub(r"\s+", " ", str(cue or "")).strip().rstrip(" .")
        # Historical/default assessment observations sometimes echo the source
        # requirement verbatim.  Treat that as no specific observation so the
        # user sees the actual visible code relationship instead of repetition.
        observation_key = cls._normalized_phrase(observation)
        cue_key = cls._normalized_phrase(cue_clean)
        if observation and observation_key and observation_key != cue_key:
            return observation

        code = cls._code_only_view(str(question or ""))
        lines = [line.strip() for line in code.splitlines() if line.strip()]
        # Prefer a concise executable/declaration line that visually exposes the
        # construct.  This is presentation only; Rule selection happened earlier.
        interesting = []
        for line in lines:
            if line in {"{", "}"} or line.endswith("{"):
                continue
            if re.match(r"^(?:case\b|default\s*:)", line, re.I):
                continue
            interesting.append(line)

        if interesting:
            shown = interesting[-1] if len(interesting) == 1 else interesting[0]
            # For common operators/control-flow, the most diagnostic line is the
            # one containing the operator rather than the first declaration.
            for marker in ("++", "--", "&&", "||", "<<", ">>", " = ", "+", "-", "if (", "if(", "for (", "for(", "return", "#define"):
                hit = next((line for line in interesting if marker in line), None)
                if hit:
                    shown = hit
                    break
            if len(shown) > 140:
                shown = shown[:137].rstrip() + "..."
            return f"The visible construct `{shown}` is the code element matched to this MISRA requirement."

        if cue_clean:
            return "The described scenario directly matches the applicability of this MISRA requirement."
        return ""

    @classmethod
    def deterministic_requirement_lookup(cls, question: str, results: Sequence[Mapping]) -> str:
        """Render matched Rule/Directive statements for a text-only concept lookup."""

        if not cls.natural_requirement_lookup_intent(question):
            return ""
        candidates = [
            item for item in (results or [])
            if isinstance(item, Mapping)
            and (
                item.get("_misra_rule_body_rescue") is True
                or bool(item.get("_structured_topic_family"))
            )
            and cls._is_citable_rule_body_record(item)
        ]
        if not candidates or not cls.supports_requested_standard(question, candidates):
            return ""

        lines = []
        seen = set()
        for item in candidates:
            if (
                item.get("_misra_rule_body_rescue") is True
                and float(item.get("_misra_cue_coverage", 0.0) or 0.0) < 0.72
            ):
                continue
            metadata = item.get("metadata", {}) or {}
            section_type = str(metadata.get("section_type", "rule") or "rule").casefold()
            identifier = str(metadata.get("directive_id", "") or metadata.get("rule_id", "") or "").strip()
            if not identifier:
                continue
            kind = "Directive" if section_type == "directive" else "Rule"
            reference = f"{kind} {identifier}"
            if reference in seen:
                continue
            statement = cls._source_rule_statement(str(item.get("text", "") or ""))
            if not statement:
                continue
            seen.add(reference)
            lines.append(f"- **{reference}:** {statement}")

        if not lines:
            return ""
        answer = lines[0][2:] if len(lines) == 1 else "\n".join(lines)
        return answer if cls.references_are_grounded(answer, candidates) else ""

    @classmethod
    def deterministic_guidance(cls, question: str, results: Sequence[Mapping]) -> str:
        """Render relevant MISRA requirements without inventing a violation verdict."""

        if not cls.guidance_intent(question):
            return ""

        candidates = [
            item for item in (results or [])
            if isinstance(item, Mapping)
            and (
                item.get("_misra_rule_body_rescue") is True
                or bool(item.get("_structured_topic_family"))
            )
            and cls._is_citable_rule_body_record(item)
        ]
        if not candidates or not cls.supports_requested_standard(question, candidates):
            return ""

        for item in candidates:
            if (
                item.get("_misra_rule_body_rescue") is True
                and float(item.get("_misra_cue_coverage", 0.0) or 0.0) < 0.72
            ):
                return ""

        clean = re.sub(r"\s+", " ", str(question or "").casefold())
        taglish = bool(re.search(
            r"\b(?:ako|ko|mga|ano|anong|alin|kailangan|bantayan|gusto|gawing|para|hindi|pwede|puwede)\b",
            clean,
        ))
        heading = "### Mga relevant na MISRA rule" if taglish else "### Relevant MISRA rules"
        lines = [heading]

        seen = set()
        for item in candidates:
            metadata = item.get("metadata", {}) or {}
            section_type = str(metadata.get("section_type", "rule") or "rule").casefold()
            identifier = str(
                metadata.get("directive_id", "")
                or metadata.get("rule_id", "")
                or ""
            ).strip()
            if not identifier:
                continue
            kind = "Directive" if section_type == "directive" else "Rule"
            reference = f"{kind} {identifier}"
            if reference in seen:
                continue
            seen.add(reference)
            statement = cls._source_rule_statement(str(item.get("text", "") or ""))
            if statement:
                lines.append(f"- **{reference}** — {statement}")

        if len(lines) == 1:
            return ""

        if taglish:
            lines.extend([
                "",
                "Guidance ito para sa planong pagbabago; kailangan pa rin ang actual code para masabi kung may violation talaga.",
            ])
        else:
            lines.extend([
                "",
                "This is guidance for the planned change; actual code is still needed to determine whether a violation exists.",
            ])

        answer = "\n".join(lines).strip()
        return answer if cls.references_are_grounded(answer, candidates) else ""

    @classmethod
    def deterministic_scope_guard_answer(
        cls,
        question: str,
        results: Sequence[Mapping],
    ) -> str:
        """Prevent one satisfied requirement from becoming an overall compliance claim."""
        clean = re.sub(r"\s+", " ", str(question or "").strip().casefold())
        if not clean or not re.search(
            r"\b(?:automatically|by\s+itself|alone|enough)\b|"
            r"\b(?:make|mean|prove|ensure|guarantee)\b.{0,30}\bmisra[- ]?compliant\b",
            clean,
        ):
            return ""
        if "misra" not in clean or "compliant" not in clean:
            return ""
        candidates = [
            item for item in (results or [])
            if isinstance(item, Mapping) and cls._is_citable_rule_body_record(item)
        ]
        if not candidates:
            return ""
        satisfied = [
            item for item in candidates
            if str(item.get("_misra_assessment_state", "") or "").casefold() == "compliant"
        ]
        if not satisfied:
            return ""
        refs = []
        for item in satisfied:
            meta = item.get("metadata", {}) or {}
            kind = "Directive" if str(meta.get("section_type", "")).casefold() == "directive" else "Rule"
            identifier = str(meta.get("directive_id", "") or meta.get("rule_id", "") or "").strip()
            if identifier:
                refs.append(f"{kind} {identifier}")
        refs = list(dict.fromkeys(refs))
        if not refs:
            return ""
        joined = ", ".join(refs)
        return (
            f"No. Satisfying {joined} does not automatically establish overall MISRA compliance. "
            f"It only shows that the stated facts satisfy that specific requirement; other applicable MISRA requirements still need to be checked."
        )

    @classmethod
    def deterministic_assessment(cls, question: str, results: Sequence[Mapping]) -> str:
        """Build a compact, source-grounded, ChatGPT-style MISRA assessment.

        Strong source-proven matches stay deterministic for speed and grounding,
        but the user-facing presentation is intentionally readable: a clear
        assessment, itemized relevant rules, per-rule status/why, and a concise
        conclusion. Exact Rule/Directive explanation requests continue to use
        the source-shaped path elsewhere and are not reformatted here.
        """
        candidates = [
            item for item in (results or [])
            if isinstance(item, Mapping)
            and (
                item.get("_misra_rule_body_rescue") is True
                or (
                    bool(item.get("_structured_topic_family"))
                    and "_misra_assessment_state" in item
                )
            )
            and str(item.get("_misra_cue", "") or "").strip()
            and cls._is_citable_rule_body_record(item)
        ]
        if not candidates:
            return ""
        if not cls.supports_requested_standard(question, candidates):
            return ""

        for item in candidates:
            coverage = float(item.get("_misra_cue_coverage", 0.0) or 0.0)
            if coverage < 0.72:
                return ""

        raw_question = str(question or "")
        has_visible_code = cls.looks_like_c_cpp(raw_question)
        asks_rule = bool(re.search(
            r"\b(?:which|what|anong?|alin(?:g)?)\b.*\brule(?:s)?\b",
            raw_question,
            re.IGNORECASE,
        ))
        states = [str(item.get("_misra_assessment_state", "violation") or "violation") for item in candidates]

        if asks_rule and not re.search(
            r"\b(?:compliant|compliance|violation|violat|allowed|permitted|valid|acceptable)\b",
            raw_question,
            re.IGNORECASE,
        ):
            assessment = (
                "Applicable MISRA requirement(s) found for the described topic."
                if not has_visible_code
                else "Applicable MISRA rule candidate(s) found for the visible construct."
            )
        elif any(state == "violation" for state in states):
            assessment = (
                "The described behavior is prohibited by the matched MISRA requirement(s)."
                if not has_visible_code
                else "Potential MISRA non-compliance detected in the visible construct."
            )
        elif any(state == "uncertain" for state in states):
            assessment = (
                "The matched MISRA requirement is relevant, but the question alone does not establish an actual code violation."
                if not has_visible_code
                else "No confirmed MISRA violation can be established from the visible information alone; one or more matched requirements need additional context."
            )
        else:
            assessment = (
                "The described behavior satisfies the matched MISRA requirement(s)."
                if not has_visible_code
                else "No MISRA non-compliance is established by the matched source requirements for the visible construct."
            )

        blocks: list[str] = ["### **Assessment**", f"**{assessment}**"]
        blocks.append(
            "### **Applicable MISRA Rule**"
            if len(candidates) == 1
            else "### **Relevant MISRA Rules**"
        )

        conclusion_items: list[str] = []
        available = cls.available_references(candidates)

        for index, item in enumerate(candidates, start=1):
            metadata = item.get("metadata", {}) or {}
            section_type = str(metadata.get("section_type", "rule") or "rule").casefold()
            identifier = str(
                metadata.get("directive_id", "")
                or metadata.get("rule_id", "")
                or ""
            ).strip()
            kind = "Directive" if section_type == "directive" else "Rule"
            reference = f"{kind} {identifier}".strip()
            text = str(item.get("text", "") or "")
            statement = cls._source_rule_statement(text)
            cue = str(item.get("_misra_cue", "") or "").strip()
            state = str(item.get("_misra_assessment_state", "violation") or "violation")
            observation = str(item.get("_misra_observation", "") or "").strip()
            category = re.sub(r"\s+", " ", cls._source_labeled_block(text, "Category")).strip()

            if state == "compliant":
                status_label = "Satisfied / no violation established"
                conclusion_label = "Satisfied"
            elif state == "uncertain":
                status_label = "Needs more context"
                conclusion_label = "Needs more context"
            else:
                status_label = "Potential non-compliance"
                conclusion_label = "Potential non-compliance"

            title = f"**{reference}" + (f" — {category}" if category else "") + "**"
            prefix = f"{index}. " if len(candidates) > 1 else ""
            lines = [prefix + title]
            lines.append(f"   - **Status:** {status_label}" if len(candidates) > 1 else f"- **Status:** {status_label}")
            if statement:
                req = f"   - **Requirement:** {statement}" if len(candidates) > 1 else f"- **Requirement:** {statement}"
                lines.append(req)
            why = cls._readable_why_it_applies(raw_question, cue, observation)
            if why:
                why_line = f"   - **Why it applies:** {why}" if len(candidates) > 1 else f"- **Why it applies:** {why}"
                lines.append(why_line)

            # If the current user supplied an actual function-like macro
            # invocation, show the exact mechanical substitution when it is
            # useful to the requested explanation.  This is computed from the
            # user's text and therefore cannot introduce invented parentheses.
            if "macro parameter" in cls._normalized_phrase(cue):
                expansions = cls._simple_function_macro_expansions(raw_question)
                for macro in expansions:
                    expanded = re.sub(r"\s+", " ", str(macro.get("expanded") or "")).strip()
                    invocation = re.sub(r"\s+", " ", str(macro.get("invocation") or "")).strip()
                    if expanded and invocation:
                        expansion_line = (
                            f"   - **Literal expansion:** `{invocation}` -> `{expanded}`"
                            if len(candidates) > 1
                            else f"- **Literal expansion:** `{invocation}` -> `{expanded}`"
                        )
                        lines.append(expansion_line)
                        break

            # Include at most one concise source detail.  Prefer an exception
            # when it explains a satisfied result; otherwise use rationale.
            detail_label = "Exception" if state == "compliant" else "Rationale"
            detail = cls._source_labeled_block(text, detail_label)
            detail = cls._first_sentences(detail, maximum=1, max_chars=360)
            if detail and detail_label == "Rationale":
                target_tokens = set(cls._lexical_tokens(" ".join((statement, cue, observation))))
                detail_tokens = set(cls._lexical_tokens(detail))
                overlap = target_tokens.intersection(detail_tokens)
                required_overlap = 2 if len(target_tokens) <= 3 else 3
                if len(overlap) < required_overlap:
                    detail = ""
            if detail and cls.cited_references(detail).issubset(available):
                label = "Source exception" if detail_label == "Exception" else "Rationale"
                detail_line = f"   - **{label}:** {detail}" if len(candidates) > 1 else f"- **{label}:** {detail}"
                lines.append(detail_line)

            rendered_rule = "\n".join(lines)
            blocks.append(rendered_rule)
            conclusion_items.append(f"- **{reference}:** {conclusion_label}")

        blocks.extend(["### **Conclusion**", "\n".join(conclusion_items)])
        answer = "\n\n".join(block for block in blocks if block).strip()
        return answer if cls.references_are_grounded(answer, candidates) else ""

    @classmethod
    def answer_focus(cls, question: str = "") -> str:
        if cls.yes_no_intent(question):
            return (
                "MISRA YES OR NO: Start with Yes or No only when the accepted "
                "MISRA evidence determines the requested polarity. If the evidence "
                "is uncertain, say Needs more context instead of guessing. Keep the "
                "support scoped to the matched Rule/Directive."
            )
        clean = re.sub(r"\s+", " ", str(question or "").casefold())
        if cls.semantic_cues(question) and re.search(
            r"\b(?:why|bakit|reason|rationale|purpose|dahilan|layunin)\b", clean
        ):
            return (
                "MISRA RATIONALE: Return the source-backed rationale for the "
                "matched Rule/Directive. Do not reuse a previous topic unless the "
                "current question contains an explicit anaphoric reference."
            )
        if cls.natural_requirement_lookup_intent(question):
            return (
                "MISRA REQUIREMENT LOOKUP: Return the matched Rule/Directive "
                "identifier and its exact source requirement. Do not turn a "
                "text-only concept lookup into a code-compliance verdict."
            )
        if cls.guidance_intent(question):
            return (
                "MISRA GUIDANCE: Give source-grounded guidance for the planned or "
                "described change. List the relevant supported Rule/Directive "
                "requirements without declaring a violation or compliance status "
                "unless actual code/evidence in the current turn establishes one. "
                "Keep the response practical and conversational."
            )
        return (
            "MISRA COMPLIANCE ASSESSMENT: Assess only against the retrieved "
            "MISRA material. Identify every relevant supported Rule/Directive, "
            "explain the code/scenario relationship, and cite only identifiers "
            "that are explicitly present in the retrieved MISRA evidence."
        )

    @staticmethod
    def requested_standard_family(question: str) -> str:
        clean = re.sub(r"\s+", " ", str(question or "").casefold())
        if re.search(r"(?:misra\s*)?c\s*\+\+|\bmisra\s+cpp\b", clean):
            return "cplusplus"
        if re.search(r"\bmisra\s+c\b", clean):
            return "c"
        return ""

    @staticmethod
    def result_is_misra(item: Mapping) -> bool:
        metadata = item.get("metadata", {}) if isinstance(item, Mapping) else {}
        labels = " ".join(
            str(metadata.get(key, "") or "")
            for key in ("file_name", "file_path", "section_title")
        ).casefold()
        return "misra" in labels

    @classmethod
    def filter_misra_results(cls, results: Sequence[Mapping]) -> list[Mapping]:
        return [item for item in (results or []) if cls.result_is_misra(item)]

    @classmethod
    def supports_requested_standard(cls, question: str, results: Sequence[Mapping]) -> bool:
        requested = cls.requested_standard_family(question)
        if not requested:
            return True
        corpus = "\n".join(
            " ".join(
                [
                    str((item.get("metadata", {}) or {}).get("file_name", "")),
                    str((item.get("metadata", {}) or {}).get("section_title", "")),
                    str(item.get("text", ""))[:800],
                ]
            )
            for item in (results or [])
        ).casefold()
        if requested == "cplusplus":
            return bool(re.search(r"misra\s*c\s*\+\+|c\s*\+\+", corpus))
        # A retrieved MISRA C rule body may not repeat the publication title.
        # Structured rule text does, however, carry explicit C-language scope
        # such as "Applies to C90, C99".  Treat that as positive evidence for
        # the C family while keeping C++ strict: a C90/C99 rule must never be
        # substituted for an explicitly requested MISRA C++ rule.
        return bool(
            re.search(
                r"\bmisra\s+c(?::|\b)|\bc\s+language\b|"
                r"\bapplies\s+to\s+c(?:90|99|11|18|23)\b|"
                r"\bc(?:90|99|11|18|23)\b",
                corpus,
            )
        )

    @staticmethod
    def _normalize_ref(kind: str, identifier: str) -> str:
        canonical_kind = "directive" if str(kind).casefold().startswith("dir") else "rule"
        return f"{canonical_kind}:{identifier}"

    @classmethod
    def available_references(cls, results: Iterable[Mapping]) -> set[str]:
        """Return only Rule/Directive identifiers proven by rule-body metadata.

        A generic appendix/table/list can contain hundreds of line-start
        ``Rule x.y`` strings.  Those are references *to* rules, not evidence
        that the retrieved chunk is the body of those rules.  Compliance
        answers may therefore cite an identifier only when the parser tagged
        the accepted chunk itself as a Rule/Directive (or supplied the same
        structured metadata explicitly).
        """

        available: set[str] = set()
        for item in results or []:
            metadata = item.get("metadata", {}) if isinstance(item, Mapping) else {}
            section_type = str(metadata.get("section_type", "") or "").casefold()
            rule_id = str(metadata.get("rule_id", "") or "").strip()
            directive_id = str(metadata.get("directive_id", "") or "").strip()
            exact = str(metadata.get("exact_reference", "") or "")

            if section_type == "rule" and rule_id:
                available.add(cls._normalize_ref("rule", rule_id))
            elif section_type == "directive" and (directive_id or rule_id):
                available.add(
                    cls._normalize_ref("directive", directive_id or rule_id)
                )

            # Trust exact-reference text only when it belongs to an accepted
            # structured Rule/Directive chunk.  Never mine identifiers from a
            # generic section, appendix, summary table, or "See also" list.
            if section_type in {"rule", "directive"}:
                for match in _REF_RE.finditer(exact):
                    available.add(
                        cls._normalize_ref(
                            match.group("kind"),
                            match.group("identifier"),
                        )
                    )

        return available

    @classmethod
    def cited_references(cls, answer: str) -> set[str]:
        return {
            cls._normalize_ref(match.group("kind"), match.group("identifier"))
            for match in _REF_RE.finditer(str(answer or ""))
        }

    @classmethod
    def references_are_grounded(cls, answer: str, results: Sequence[Mapping]) -> bool:
        cited = cls.cited_references(answer)
        if not cited:
            return False
        available = cls.available_references(results)
        return bool(available) and cited.issubset(available)
