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

    def _build_prompt(
        self,
        context,
        question,
        history
    ):

        # Insert context and question into prompt template
        user_prompt = (
            ANSWER_TEMPLATE.format(
                context=context,
                history=history,
                question=question
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

        return any(
            indicator in clean
            for indicator in indicators
        )

    def _verify_multi_answer(
        self,
        context: str,
        question: str,
        draft_answer: str
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

DRAFT ANSWER:
{draft_answer}

Task:
1. Check every COMPANY KNOWLEDGE section from first to last.
2. Find every item explicitly stated that is relevant to the USER QUESTION.
3. Compare those items with the DRAFT ANSWER.
4. If the DRAFT ANSWER is complete, return the DRAFT ANSWER exactly.
5. If the DRAFT ANSWER missed relevant items, return a corrected final answer.
6. Use bullets for multiple items.
7. Return only the final answer.
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

        prompt = self._build_prompt(
            context,
            final_question,
            history
        )

        if DEBUG_MODE:

            print(
                f"\nFinal Answer Question : "
                f"{final_question}"
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

            if self._is_multi_answer_question(
                final_question
            ):

                answer = self._verify_multi_answer(
                    context=context,
                    question=final_question,
                    draft_answer=answer
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