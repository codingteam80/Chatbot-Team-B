from pathlib import Path
import re

from llm.ollama_client import OllamaClient
from services.query_service import QueryService

from config.prompts import (
    SYSTEM_PROMPT,
    ANSWER_TEMPLATE,
    NO_RESULT_MESSAGE,
    REWRITE_QUERY_PROMPT
)

from config.settings import DEBUG_MODE
from chat.chat_manager import ChatManager
from chat.query_normalizer import QueryNormalizer
from chat.conversation_resolver import ConversationResolver
from chat.query_enricher import QueryEnricher


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
            r"^(?:what is|what was|define|explain|describe)\b",
            clean
        ):

            return (
                "DEFINITION OR DETAIL: Return the direct definition, "
                "explanation, rule, requirement, behavior, configuration, "
                "or requested detail."
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
12. If the draft already has the correct focus and facts, return it unchanged.
13. If the draft has the wrong focus, return a corrected direct answer using
    only explicitly stated COMPANY KNOWLEDGE.
14. Return a complete grammatical answer.
15. Correct singular/plural agreement when the factual meaning stays unchanged.
16. If the requested focused answer is not explicitly supported, return exactly:
{fallback}
17. Return only the final answer.
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

        cleaned = self._strip_output_wrappers(
            answer
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
        Remove cases where the LLM repeats the user's question
        as the first line of the answer.
        """

        if not answer or not question:

            return answer

        question_clean = question.strip()

        if not question_clean:

            return answer

        def normalize_line(text):

            text = text.strip().lower()

            text = re.sub(
                r"[^\w\s]",
                "",
                text
            )

            text = re.sub(
                r"\s+",
                " ",
                text
            ).strip()

            return text

        normalized_question = normalize_line(
            question_clean
        )

        lines = answer.splitlines()

        cleaned_lines = []

        for index, line in enumerate(lines):

            normalized_line = normalize_line(
                line
            )

            # Remove only if the repeated question appears
            # at the beginning of the answer.
            if (
                index <= 1
                and normalized_line == normalized_question
            ):

                continue

            cleaned_lines.append(
                line
            )

        return "\n".join(
            cleaned_lines
        ).strip()

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
        search_question = (
            self.query_enricher.enrich(
                resolved_question,
                intent_question=question
            )
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

            # Correct cases where a true fact answers the wrong
            # type of question.
            answer = self._verify_answer_focus(
                context=context,
                question=final_question,
                draft_answer=answer,
                answer_focus=answer_focus,
                resolved_question=resolved_question
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

            # Final deterministic safety gate.
            # This must run after every LLM generation and verification pass.
            answer = self._apply_output_safety_gate(
                answer,
                final_question
            )

        except Exception:

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

        return {

            "answer": answer,

            "sources": sources,

            "chunks": results
        }