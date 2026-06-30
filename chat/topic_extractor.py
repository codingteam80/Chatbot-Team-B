import re


class TopicExtractor:
    """
    Extracts the latest valid conversation topic.

    A valid topic is typically a new entity such as:

    - Jose Rizal
    - Andres Bonifacio
    - ISO 9001
    - Employee Handbook

    Follow-up questions such as:

    - Where was he born?
    - What was his occupation?
    - Who was his mother?

    do NOT create a new topic.
    """

    # Common question prefixes
    QUESTION_PREFIXES = [
        r"who is",
        r"who was",
        r"who are",
        r"what is",
        r"what was",
        r"what are",
        r"where is",
        r"where was",
        r"when is",
        r"when was",
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

    # Words that indicate the question is only
    # referring to the previous topic.
    INVALID_START_WORDS = {
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "they",
        "them",
        "their",
        "theirs",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those"
    }

    # Commands that continue the current topic
    # instead of introducing a new one.
    FOLLOW_UP_COMMANDS = {
        "continue",
        "continue please",
        "tell me more",
        "tell me more about it",
        "explain more",
        "elaborate",
        "go on",
        "more",
        "keep going",
        "next"
    }

    def extract(
        self,
        history_messages
    ):
        """
        Extract the latest valid topic
        from previous user messages.
        """

        if not history_messages:
            return None

        for message in reversed(history_messages):

            if message["role"] != "user":
                continue

            topic = self._extract_from_question(
                message["content"]
            )

            if topic:
                return topic

        return None

    def extract_new_topic(
        self,
        question
    ):
        """
        Extract a topic from the CURRENT user question.

        Returns None if the question is
        only a follow-up question.
        """

        return self._extract_from_question(
            question
        )

    def _extract_from_question(
        self,
        question
    ):

        text = question.strip()

        # Remove common question prefixes.
        for prefix in self.QUESTION_PREFIXES:

            text = re.sub(
                rf"^{prefix}\s+",
                "",
                text,
                flags=re.IGNORECASE
            )

        # Remove ending punctuation.
        text = re.sub(
            r"[?!.]+$",
            "",
            text
        ).strip()

        if not text:
            return None

        # Ignore follow-up commands.
        if text.lower() in self.FOLLOW_UP_COMMANDS:
            return None

        words = text.split()

        if not words:
            return None

        first_word = words[0].lower()

        # Ignore follow-up questions.
        if first_word in self.INVALID_START_WORDS:
            return None

        # Ignore incomplete question phrases such as:
        #
        # What it did
        # What this means
        # Where it happened
        # How it works
        #
        # These refer to the previous topic and
        # should NOT become a new topic.
        QUESTION_WORDS = {
            "what",
            "where",
            "when",
            "why",
            "how",
            "which"
        }

        if (
            first_word in QUESTION_WORDS
            and len(words) > 1
        ):
            second_word = words[1].lower()

            if (
                second_word in self.INVALID_START_WORDS
                or second_word in {
                    "did",
                    "does",
                    "do",
                    "is",
                    "are",
                    "was",
                    "were",
                    "can",
                    "could",
                    "should",
                    "would",
                    "will"
                }
            ):
                return None

        return text