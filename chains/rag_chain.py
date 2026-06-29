from langchain_core.prompts import ChatPromptTemplate


class RAGChain:

    @staticmethod
    def build_prompt():

        return ChatPromptTemplate.from_template(
            """
You are a company knowledge assistant.

Rules:
- Answer ONLY using the provided context.
- If information is missing, say you cannot find it.
- Do not hallucinate.
- Be concise and professional.

Context:
{context}

Question:
{question}

Answer:
"""
        )

    @staticmethod
    def format_prompt(
        context: str,
        question: str
    ):

        prompt = RAGChain.build_prompt()

        return prompt.format_messages(
            context=context,
            question=question
        )