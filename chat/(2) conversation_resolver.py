import re


class ConversationResolver:
    """
    Resolves follow-up questions into standalone questions.

    This class does NOT use an LLM.

    It simply looks at the recent conversation and
    replaces pronouns with the latest discussed topic.

    Examples

    User:
    Tell me about Jose Rizal.

    User:
    Where was he born?

    →

    Where was Jose Rizal born?
    """

    # Pronouns we want to resolve
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

    def resolve(
        self,
        history_messages,
        question
    ):
        """
        Returns a rewritten standalone question.
        """

        if not history_messages:
            return question

        topic = self._find_latest_topic(
            history_messages
        )

        if not topic:
            return question

        return self._replace_pronouns(
            question,
            topic
        )

    def _find_latest_topic(
        self,
        history_messages
    ):
        """
        Very simple heuristic.

        Find the latest capitalized phrase from
        previous conversation.

        Example:

        Emilio Aguinaldo

        Treaty of Paris

        Battle of Mactan
        """

        text = ""

        for message in reversed(history_messages):

            text += (
                message["content"]
                + "\n"
            )

            matches = re.findall(

                r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5})",

                text
            )

            if matches:

                return matches[0]

        return None

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