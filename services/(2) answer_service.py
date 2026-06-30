from llm.ollama_client import (
    OllamaClient
)

from services.query_service import (
    QueryService
)

from config.prompts import (
    SYSTEM_PROMPT,
    ANSWER_TEMPLATE,
    NO_RESULT_MESSAGE,
    REWRITE_QUERY_PROMPT
)

from config.settings import DEBUG_MODE

from chat.chat_manager import ChatManager

class AnswerService:

    def __init__(self):

        # Retrieve relevant company knowledge
        self.query_service = (
            QueryService()
        )

        # Generate answers using Ollama
        self.llm = OllamaClient()

    def _build_chat_history(self):

        """
        Build conversation history from the
        current chat session.

        Only include the latest messages to
        avoid sending an excessively long prompt.
        """

        messages = (
            ChatManager.get_current_messages()
        )

        # Keep only the last 6 messages
        messages = messages[-6:]

        history = []

        for message in messages:

            role = (
                message["role"]
                .capitalize()
            )

            content = (
                message["content"]
            )

            history.append(
                f"{role}: {content}"
            )

        return "\n".join(history)

    def _rewrite_question(
        self,
        question,
        history
    ):

        """
        Rewrite follow-up questions into
        standalone questions before retrieval.
        """

        # No history → use original question
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

        # Store unique source filenames
        source_files = []

        for item in results:

            # Get source file name from metadata
            file_name = (
                item["metadata"]
                .get(
                    "file_name",
                    "Unknown"
                )
            )

            # Prevent duplicate source names
            if file_name not in source_files:

                source_files.append(
                    file_name
                )

        return source_files

    def _postprocess_answer(
        self,
        answer: str
    ):

        """
        Clean the LLM response before returning it.
        """

        answer = answer.strip()

        fallback = NO_RESULT_MESSAGE.strip()

        # If the model returned only the fallback
        if answer == fallback:
            return fallback

        # Remove accidental fallback appended
        answer = answer.replace(
            fallback,
            ""
        )

        # Remove excessive blank lines
        while "\n\n\n" in answer:

            answer = answer.replace(
                "\n\n\n",
                "\n\n"
            )

        return answer.strip()

    def ask(
        self,
        question
    ):

        history = (
            self._build_chat_history()
        )

        #search_question = (
        #     self._rewrite_question(
        #         question,
        #         history
        #     )
        #)

        search_question = question

        # ======================================
        # DEBUG
        # ======================================
        if DEBUG_MODE:

            print("=" * 60)
            print("Conversation Debug")
            print("=" * 60)

            print(f"Original Question : {question}")
            print(f"Search Question   : {search_question}")

            print("\nHistory:")

            print(history)

            print("=" * 60)

        # Retrieve context from RAG pipeline
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

            print(f"Retrieved Chunks : {len(results)}")

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

        # Return fallback message if nothing found
        if not context:

            return {

                "answer":
                    NO_RESULT_MESSAGE,

                "sources": [],

                "chunks": []
            }

        # Build final prompt for LLM
        prompt = self._build_prompt(
            context,
            question,
            history
        )

        if DEBUG_MODE:
            print("\nFinal Prompt:\n")
            print(prompt)
            print("=" * 60)

        # Generate answer from Ollama
        try:

            answer = (
                self.llm.generate(
                    prompt
                )
            )

            # Clean the response
            answer = self._postprocess_answer(
                answer
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

        # Extract source file names
        sources = (
            self._extract_sources(
                results
            )
        )

        # Return answer, sources, and retrieved chunks
        return {

            "answer": answer,

            "sources": sources,

            "chunks": results
        }