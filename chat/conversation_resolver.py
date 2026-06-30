import re

from chat.topic_extractor import (
    TopicExtractor
)


class ConversationResolver:
    """
    Resolves follow-up questions into standalone questions.

    Example

    User:
    Who is Jose Rizal?

    User:
    Where was he born?

    →

    Where was Jose Rizal born?
    """

    # Subject pronouns
    SUBJECT_PRONOUNS = {

        "he",
        "she",
        "him",

        "they",
        "them",

        "it",

        "this",
        "that",
        "these",
        "those"
    }

    # Possessive pronouns
    POSSESSIVE_PRONOUNS = {

        "his",
        "her",
        "hers",

        "their",
        "theirs",

        "its"
    }

    # Common follow-up commands
    FOLLOW_UP_PHRASES = {

        "continue",
        "tell me more",
        "explain more",
        "elaborate",
        "more",
        "go on"
    }

    def __init__(self):

        self.topic_extractor = (
            TopicExtractor()
        )

    def resolve(
        self,
        history_messages,
        question
    ):

        if not history_messages:

            return question

        topic = (
            self.topic_extractor.extract(
                history_messages
            )
        )

        if not topic:

            return question

        # Handle commands like:
        # Continue
        # Tell me more
        # Explain more
        followup = self._handle_followup_command(
            question,
            topic
        )

        if followup != question:

            return followup

        return self._replace_pronouns(
            question,
            topic
        )

    def _handle_followup_command(
        self,
        question,
        topic
    ):

        clean = (
            question
            .lower()
            .strip()
            .rstrip("?.!")
        )

        if clean in self.FOLLOW_UP_PHRASES:

            return f"{question.strip()} about {topic}"

        return question

    def _replace_pronouns(
        self,
        question,
        topic
    ):

        words = question.split()

        new_words = []

        replaced = False

        for word in words:

            punctuation = ""

            if word and word[-1] in ".,?!":

                punctuation = word[-1]

            clean = (
                word.lower()
                .strip(".,?!")
            )

            # Subject pronouns
            if (
                clean in self.SUBJECT_PRONOUNS
                and not replaced
            ):

                new_words.append(
                    topic + punctuation
                )

                replaced = True

                continue

            # Possessive pronouns
            if (
                clean in self.POSSESSIVE_PRONOUNS
                and not replaced
            ):

                new_words.append(
                    topic + "'s" + punctuation
                )

                replaced = True

                continue

            new_words.append(word)

        return " ".join(new_words)