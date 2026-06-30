from llm.ollama_client import (
    OllamaClient
)

from services.query_service import (
    QueryService
)

from config.prompts import (
    SYSTEM_PROMPT,
    ANSWER_TEMPLATE,
    NO_RESULT_MESSAGE
)


class AnswerService:

    def __init__(self):

        self.query_service = (
            QueryService()
        )

        self.llm = OllamaClient()

    def _build_prompt(
        self,
        context,
        question
    ):

        user_prompt = (
            ANSWER_TEMPLATE.format(
                context=context,
                question=question
            )
        )

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

        source_files = []

        for item in results:

            file_name = (
                item["metadata"]
                .get(
                    "file_name",
                    "Unknown"
                )
            )

            if file_name not in source_files:

                source_files.append(
                    file_name
                )

        return source_files

    def ask(
        self,
        question
    ):

        context, results = (
            self.query_service
            .retrieve_context(
                question
            )
        )

        if not context:

            return {

                "answer":
                    NO_RESULT_MESSAGE,

                "sources": []
            }

        prompt = self._build_prompt(
            context,
            question
        )

        # ==================================================
        # DEBUG OUTPUT
        # ==================================================

        print("\n" + "=" * 80)
        print("QUESTION")
        print("=" * 80)
        print(question)

        print("\n" + "=" * 80)
        print("CONTEXT")
        print("=" * 80)
        print(context[:5000])

        print("\n" + "=" * 80)
        print("PROMPT")
        print("=" * 80)
        print(prompt[:5000])

        print("\n" + "=" * 80)
        print("TOP RETRIEVED CHUNKS")
        print("=" * 80)

        for i, item in enumerate(results[:5], start=1):

            print(f"\n[{i}]")

            print(
                "FILE:",
                item["metadata"].get(
                    "file_name",
                    "Unknown"
                )
            )

            print(
                "TEXT:"
            )

            print(
                item["text"][:500]
            )

        print("=" * 80)

        # ==================================================
        # LLM CALL
        # ==================================================

        answer = (
            self.llm.generate(
                prompt
            )
        )

        sources = (
            self._extract_sources(
                results
            )
        )

        return {

            "answer": answer,

            "sources": sources,

            "chunks": results
        }