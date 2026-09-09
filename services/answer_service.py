from pathlib import Path
import re

from llm.ollama_client import OllamaClient
from services.query_service import QueryService

from config.prompts import (
    SYSTEM_PROMPT,
    ANSWER_TEMPLATE,
    NO_RESULT_MESSAGE,
    REWRITE_QUERY_PROMPT,
    MULTILINGUAL_RETRIEVAL_QUERY_PROMPT
)

from config.settings import (
    DEBUG_MODE,
    ENABLE_MULTILINGUAL_RETRIEVAL,
    MULTILINGUAL_RETRY_ON_EMPTY,
    MULTILINGUAL_QUERY_MAX_CHARS
)
from chat.chat_manager import ChatManager
from chat.query_normalizer import QueryNormalizer
from chat.conversation_resolver import ConversationResolver
from chat.query_enricher import QueryEnricher
from qa.evidence_logger import evidence_logger
from utils.structured_reference import extract_structured_reference


class AnswerService:

    def __init__(self):

        # Retrieve relevant company knowledge
        self.query_service = (
            QueryService()
        )

        # Generate answers using Ollama
        self.llm = OllamaClient()

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

            rewritten = self.llm.generate(
                prompt
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
        Detect questions that contain two or more requested parts.

        This is generic and supports common English and Tagalog
        interrogative forms without depending on a document topic.
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

        # Two explicit interrogative clauses joined by "and" or "at".
        if re.search(
            interrogatives
            + r".+\b(?:and|at)\b.+"
            + interrogatives,
            clean
        ):

            return True

        # A first interrogative clause followed by a second requested
        # noun phrase, such as:
        # "When ... and what event ...?"
        # "Kailan ... at anong pangyayari ...?"
        if re.search(
            r"^"
            + interrogatives
            + r".+\b(?:and|at)\b\s+"
            + interrogatives,
            clean
        ):

            return True

        # Multiple questions separated by punctuation.
        if len(
            re.findall(
                r"[?;]",
                question
            )
        ) >= 2:

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

        structured_reference = extract_structured_reference(
            resolved_question or question
        )

        if structured_reference:

            if re.search(
                r"^(?:explain|describe)\b",
                clean
            ):

                return (
                    "STRUCTURED EXPLANATION: Explain the exact requested "
                    "Rule, Directive, or Section using the important supported "
                    "content tied to that same identifier. For a Rule or "
                    "Directive, state the exact requirement first, then cover "
                    "relevant amplification or scope and the rationale when "
                    "available. Keep the explanation complete enough to "
                    "understand the requested item without adding unsupported "
                    "information, unrelated neighboring identifiers, or "
                    "cross-reference content as a substitute for the target."
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
                )
            ):

                return (
                    "STRUCTURED STATEMENT: Return the exact requested Rule "
                    "or Directive statement itself. The first direct "
                    "requirement/title line has priority over rationale, "
                    "examples, cross-references, or applicability details."
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

        # Explicit question form has priority over words appearing later.
        if re.search(
            r"^how\s+(?:many|much)\b",
            clean
        ):

            return (
                "QUANTITY: Return the requested number, amount, duration, "
                "size, limit, count, or value with its unit when available."
                + target_instruction
            )

        # Approval has priority over generic authorization.
        if (
            re.search(
                r"^who\s+(?:can\s+|may\s+)?"
                r"(?:approve|approves|approved|authorize|authorizes|authorized)\b",
                clean
            )
            or re.search(
                r"\b(?:approver|approvers|approval authority|"
                r"approval authorities)\b",
                clean
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
                r"\b(?:authorized|authorization|allowed|permitted|"
                r"permission)\b",
                clean
            )
        ):

            return (
                "AUTHORIZED ENTITY: Return the person, group, role, team, "
                "organization, system, component, or entity allowed to act."
                + target_instruction
            )

        if re.search(
            r"\b(?:eligible|eligibility|qualified|qualification|"
            r"entitled|entitlement)\b",
            clean
        ):

            return (
                "ELIGIBLE OR ENTITLED ENTITY: Return the person, group, role, "
                "category, organization, system, component, or entity that is "
                "eligible, qualified, or entitled. Do not substitute a "
                "quantity, time, location, reason, or procedure."
                + target_instruction
            )

        if re.search(
            r"^when\b",
            clean
        ):

            return (
                "TIME: Return the date, time, schedule, period, deadline, "
                "frequency, sequence point, or triggering condition."
                + target_instruction
                + " Do not substitute another date or time merely because "
                "it appears in the same context."
            )

        if re.search(
            r"^where\b",
            clean
        ):

            return (
                "LOCATION: Return the requested place, path, section, module, "
                "system area, storage location, interface, or position."
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
        if re.search(
            r"^who\s+(?:is|was)\b",
            clean
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
            r"^(?:what is|what was|define)\b",
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
            "YES OR NO:",
        )

        return answer_focus.startswith(
            prefixes
        )

    def _verify_answer_focus(
        self,
        context: str,
        question: str,
        draft_answer: str,
        answer_focus: str,
        resolved_question: str = ""
    ):

        """
        Correct a true but wrongly focused draft using only
        the retrieved company knowledge.
        """

        fallback = NO_RESULT_MESSAGE.strip()

        if (
            not context
            or not draft_answer
            or draft_answer.strip().lower()
            == fallback.lower()
        ):

            return draft_answer

        if not self._should_verify_answer_focus(
            answer_focus
        ):

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

            verifier_output = self.llm.generate(
                verify_prompt
            )

            verified_answer = (
                self._extract_verifier_answer(
                    verifier_output,
                    draft_answer
                )
            )

            if not verified_answer:

                return draft_answer

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
        results
    ):

        sources = []

        for item in results:

            metadata = item["metadata"]

            source = {

                "name": metadata.get(
                    "file_name",
                    "Unknown"
                ),

                "path": metadata.get(
                    "file_path",
                    ""
                )
            }

            if source not in sources:

                sources.append(source)

        # Display only the highest-ranked source.
        return sources[:1]

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
            r"(?im)^\s*(?:structured\s+statement|structured\s+explanation|"
            r"grounded\s+explanation|definition\s+or\s+detail|identity\s+or\s+overview|"
            r"short\s+topic\s+overview|reason|procedure|list|time|location|"
            r"quantity|entity\s+or\s+choice|person\s+or\s+entity|"
            r"authorized\s+entity|responsible\s+entity|approver|"
            r"eligible\s+or\s+entitled\s+entity|compound|general)\s*:\s*",
            "",
            answer,
            count=1
        )

        answer = re.sub(
            r"(?im)^\s*```(?:markdown|text)?\s*$",
            "",
            answer
        )

        answer = re.sub(
            r"(?im)^\s*```\s*$",
            "",
            answer
        )

        return answer.strip()

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
            r"(?ims)^\s*Rationale\s*$\s*(.+?)(?=^\s*(?:Amplification|Example|"
            r"Exception|See\s+also|Category|Analysis|Applies\s+to|Rule\s+\d|"
            r"Dir(?:ective)?\s+\d|Section\s+\d|={3,})\b|\Z)",
            context
        )

        if not match:
            return ""

        rationale = re.sub(
            r"\s+",
            " ",
            match.group(1)
        ).strip()

        if not rationale:
            return ""

        sentence_match = re.match(
            r"(.+?[.!?])(?:\s|$)",
            rationale
        )

        if sentence_match:
            return sentence_match.group(1).strip()

        return rationale

    def _structured_labeled_block_from_context(
        self,
        context: str,
        label: str
    ):

        """Return one named subsection from an exact structured block.

        This keeps explanation enrichment deterministic and source-grounded.
        It never searches outside the already selected exact structured context.
        """

        if not context or not label:
            return ""

        boundary = (
            r"(?:Category|Analysis|Applies\s+to|Rationale|Amplification|"
            r"Example|Examples|Exception|Exceptions|See\s+also|Notes?|"
            r"Rule\s+\d|Dir(?:ective)?\s+\d|Section\s+\d|={3,})"
        )

        match = re.search(
            rf"(?ims)^\s*{re.escape(label)}\s*$\s*(.+?)(?=^\s*{boundary}\b|\Z)",
            context
        )

        if not match:
            return ""

        return re.sub(
            r"\s+",
            " ",
            match.group(1)
        ).strip()

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

    def _structured_explanation_from_context(
        self,
        context: str,
        reference,
        display_name: str
    ):

        """Build a fuller exact Rule/Directive explanation from its own block."""

        statement = self._structured_statement_from_context(
            context,
            reference
        )

        if not statement:
            return ""

        pieces = [
            f"{display_name}: {statement.rstrip('.')}."
        ]

        amplification = self._structured_labeled_block_from_context(
            context,
            "Amplification"
        )

        if amplification:
            pieces.append(
                self._limit_grounded_explanation_text(
                    amplification,
                    max_sentences=2,
                    max_chars=550
                )
            )

        rationale = self._structured_labeled_block_from_context(
            context,
            "Rationale"
        )

        if rationale:
            pieces.append(
                self._limit_grounded_explanation_text(
                    rationale,
                    max_sentences=3,
                    max_chars=900
                )
            )

        return " ".join(
            piece
            for piece in pieces
            if piece
        ).strip()

    def _section_heading_and_intro_from_context(
        self,
        context: str,
        reference
    ):

        """Return the exact Section title and its opening explanation."""

        if (
            not context
            or reference is None
            or reference.kind != "section"
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
            rf"^(?:Section\s+)?{identifier}\s*(?::|[-–—])?\s*(.*)$",
            re.IGNORECASE
        )
        subsection_re = re.compile(
            rf"^{identifier}\.\d+(?:\s+.*)?$",
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
            or reference.kind != "section"
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
            rf"^{identifier}\.(\d+)(?:\s+(.*))?$",
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
            or reference.kind != "section"
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
            rf"^{identifier}\.(\d+)(?:\s+(.*))?$",
            re.IGNORECASE
        )
        repeated_section_re = re.compile(
            rf"^Section\s+{identifier}\b",
            re.IGNORECASE
        )
        any_subsection_re = re.compile(
            rf"^{identifier}\.\d+(?:\.\d+)*\b",
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

        if not title and not intro and not topics:
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
            for topic_title, key_text in topic_summaries:
                cleaned_key_text = (key_text or "").strip()
                if (
                    cleaned_key_text.endswith(".")
                    and not cleaned_key_text.endswith("...")
                ):
                    cleaned_key_text = cleaned_key_text[:-1]

                if cleaned_key_text:
                    detail_items.append(
                        f"- {topic_title}: {cleaned_key_text}"
                    )
                else:
                    detail_items.append(f"- {topic_title}")

            pieces.append(
                "Key points:\n"
                + "\n".join(detail_items)
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

        prose_pieces = []
        bullet_piece = ""

        for piece in pieces:
            if not piece:
                continue
            if "\n- " in piece:
                bullet_piece = piece.strip()
            else:
                prose_pieces.append(piece.rstrip("."))

        answer = ". ".join(prose_pieces).strip()

        if answer and not answer.endswith("."):
            answer += "."

        if bullet_piece:
            answer = (answer + "\n\n" + bullet_piece).strip()

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

        if reference.kind == "section":
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

        display_name = (
            f"Directive {reference.identifier}"
            if reference.kind == "directive"
            and re.search(r"\bdirective\b", question or "", re.IGNORECASE)
            else reference.display_name
        )

        if answer_focus.startswith("STRUCTURED STATEMENT:"):
            return f"{display_name}: {statement}"

        rationale = self._structured_rationale_from_context(
            context
        )

        if answer_focus.startswith("STRUCTURED EXPLANATION:"):
            explanation = self._structured_explanation_from_context(
                context=context,
                reference=reference,
                display_name=display_name
            )

            if explanation:
                return explanation

            return f"{display_name}: {statement}"

        if answer_focus.startswith("REASON:") and rationale:
            return rationale

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

        return self._looks_like_plural_list_question(
            question
        )

    def _verify_multi_answer(
        self,
        context: str,
        question: str,
        draft_answer: str,
        resolved_question: str = ""
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

            verified_answer = self.llm.generate(
                verify_prompt
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

    def _verify_compound_answer(
        self,
        context: str,
        question: str,
        draft_answer: str,
        resolved_question: str = ""
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

        verify_prompt = f"""
Produce the final answer to a multi-part question.

Use ONLY the COMPANY KNOWLEDGE below.
Do not use outside knowledge.
Do not guess.
Do not invent facts.
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
2. Answer every part in the original order.
3. Do not omit a date, event, reason, entity, quantity, location, procedure,
   comparison, or explanation requested by another clause.
4. Use only facts explicitly supported by COMPANY KNOWLEDGE.
5. Use short bullets when separate parts are clearer as separate items.
6. If one requested part is not supported, state the exact fallback only for
   that unsupported part; do not invent an answer.
7. Return only the final user-facing answer.
"""

        try:

            verified_answer = self.llm.generate(
                verify_prompt
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

        file_name = (
            metadata
            .get(
                "file_name",
                ""
            )
            .strip()
        )

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

        if (
            first_word_match
            and first_word_match.group(0)
            in english_starters
        ):

            return True

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

        return bool(
            words.intersection(
                english_function_words
            )
        )

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

            english_query = self.llm.generate(
                prompt
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


    def ask(
        self,
        question
    ):

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

        # ======================================
        # Resolve Follow-up References
        # ======================================
        resolved_question = (
            self.conversation_resolver.resolve(
                previous_messages,
                normalized_question
            )
        )

        # ======================================
        # Enrich Question for Retrieval Only
        # ======================================
        original_search_question = (
            self.query_enricher.enrich(
                resolved_question,
                intent_question=question
            )
        )

        english_retrieval_query = ""

        if (
            ENABLE_MULTILINGUAL_RETRIEVAL
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
        context, results = (
            self.query_service
            .retrieve_context(
                search_question
            )
        )

        # A bilingual query may occasionally dilute reranker relevance.
        # When it returns no accepted context, retry using only the
        # canonical English query. This keeps English performance stable
        # while allowing non-English questions to retrieve English documents.
        if (
            not context
            and MULTILINGUAL_RETRY_ON_EMPTY
            and english_search_question
            and english_search_question.lower()
            != search_question.lower()
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

            context, results = (
                self.query_service
                .retrieve_context(
                    english_search_question
                )
            )

            if context:

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
        # Build Final Prompt
        # ======================================
        # Use the original user question for the final prompt
        # so the LLM preserves the user's language.
        #
        # Retrieval uses search_question.
        # Answer generation uses final_question.
        final_question = question.strip()

        answer_focus = (
            self._detect_answer_focus(
                final_question,
                resolved_question
            )
        )

        evidence_logger.record_event(
            event_name="ANSWER FOCUS",
            details=answer_focus,
            status="DETECTED"
        )

        prompt = self._build_prompt(
            context,
            final_question,
            history,
            answer_focus
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
            
            print("\nFinal Prompt:\n")
            print(prompt)
            print("=" * 60)

        # ======================================
        # Generate Answer
        # ======================================
        draft_answer = ""
        verified_answer = ""
        prompt_leak_detected = False
        generation_error = ""

        try:

            answer = (
                self.llm.generate(
                    prompt
                )
            )

            answer = self._postprocess_answer(
                answer,
                final_question
            )

            draft_answer = answer

            # Correct cases where a true fact answers the wrong
            # type of question.
            answer = self._verify_answer_focus(
                context=context,
                question=final_question,
                draft_answer=answer,
                answer_focus=answer_focus,
                resolved_question=resolved_question
            )

            answer = self._apply_structured_answer_focus(
                context=context,
                question=final_question,
                resolved_question=resolved_question,
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
                    resolved_question=resolved_question
                )

            if self._is_compound_question(
                final_question
            ):

                answer = self._verify_compound_answer(
                    context=context,
                    question=final_question,
                    draft_answer=answer,
                    resolved_question=resolved_question
                )

            answer = self._relation_aware_date_answer(
                context=context,
                question=final_question,
                resolved_question=resolved_question,
                answer_focus=answer_focus,
                current_answer=answer
            )

            answer = self._remove_invalid_yes_no_prefix(
                answer,
                answer_focus
            )

            verified_answer = answer

            # Final deterministic safety gate.
            # This must run after every LLM generation and verification pass.
            prompt_leak_detected = (
                self._contains_prompt_leak(
                    answer
                )
            )

            answer = self._apply_output_safety_gate(
                answer,
                final_question
            )

        except Exception as error:

            generation_error = str(
                error
            )

            evidence_logger.record_error(
                location="AnswerService.ask",
                error=error,
                details={
                    "question":
                        final_question,

                    "resolved_question":
                        resolved_question,
                }
            )

            answer = (
                "An error occurred while generating the answer."
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

            evidence_logger.record_answer(
                draft_answer=draft_answer,
                verified_answer=verified_answer,
                final_answer=NO_RESULT_MESSAGE,
                fallback_used=True,
                prompt_leak_detected=prompt_leak_detected,
                sources=[],
                error=generation_error
            )

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


        # Extract sources only for a valid answer.
        sources = self._extract_sources(
            results
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

        return {

            "answer": answer,

            "sources": sources,

            "chunks": results
        }