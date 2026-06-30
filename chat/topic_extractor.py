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
        r"what is",
        r"what was",
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

    # If a question starts with one of these after
    # removing the prefix, it is probably a follow-up
    # question rather than a new topic.
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

    def extract(
        self,
        history_messages
    ):

        if not history_messages:
            return None

        # Look at previous user questions only.
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

        words = text.split()

        if not words:
            return None

        first_word = words[0].lower()

        # Examples:
        #
        # his occupation
        # her mother
        # he born
        #
        # These are follow-up questions,
        # not new conversation topics.
        if first_word in self.INVALID_START_WORDS:
            return None

        return text