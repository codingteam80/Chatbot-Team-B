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

    PRONOUNS = {

        "he",
        "she",
        "him",
        "his",
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

        return self._replace_pronouns(
            question,
            topic
        )

    def _replace_pronouns(
        self,
        question,
        topic
    ):

        words = question.split()

        new_words = []

        replaced = False

        for word in words:

            clean = (
                word.lower()
                .strip(".,?!")
            )

            if (
                clean in self.PRONOUNS
                and not replaced
            ):

                new_words.append(topic)

                replaced = True

            else:

                new_words.append(word)

        return " ".join(new_words)