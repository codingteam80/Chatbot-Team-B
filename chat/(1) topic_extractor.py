import re


class TopicExtractor:
    """
    Extracts the main topic from the latest
    user message.

    This class is intentionally simple and
    does not use any LLM.
    """

    # Common question prefixes
    QUESTION_PREFIXES = [
        r"who is",
        r"what is",
        r"where is",
        r"when is",
        r"when was",
        r"where was",
        r"tell me about",
        r"describe",
        r"explain",
        r"give me information about",
        r"give information about",
        r"can you explain",
        r"can you tell me about",
        r"please explain",
        r"define",
        r"introduce",
        r"summarize"
    ]

    def extract(
        self,
        history_messages
    ):

        if not history_messages:

            return None

        # Look only at previous user messages.
        for message in reversed(history_messages):

            if message["role"] != "user":
                continue

            topic = self._extract_from_question(
                message["content"]
            )

            if topic:

                return topic

        return None

    def _extract_from_question(
        self,
        question
    ):

        text = question.strip()

        # Remove common prefixes.
        for prefix in self.QUESTION_PREFIXES:

            text = re.sub(

                rf"^{prefix}\s+",

                "",

                text,

                flags=re.IGNORECASE
            )

        # Remove punctuation.
        text = re.sub(

            r"[?!.]+$",

            "",

            text
        )

        return text.strip()