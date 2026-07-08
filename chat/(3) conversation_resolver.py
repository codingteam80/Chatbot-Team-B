from chat.topic_extractor import TopicExtractor
from chat.chat_manager import ChatManager


class ConversationResolver:
    """
    Resolves follow-up questions using the current conversation topic.

    Example:
        Current topic: José Rizal
        User: where was he born?
        Output: where was José Rizal born?
    """

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

    POSSESSIVE_PRONOUNS = {
        "his",
        "her",
        "hers",
        "their",
        "theirs",
        "its"
    }

    FOLLOW_UP_PHRASES = {
        "continue",
        "tell me more",
        "explain more",
        "elaborate",
        "more",
        "go on"
    }

    def __init__(self):

        self.topic_extractor = TopicExtractor()

    def resolve(
        self,
        history_messages,
        question
    ):

        if not question:

            return question

        # ======================================
        # If the question contains pronouns,
        # do NOT extract a new topic.
        #
        # Use current topic memory instead.
        # ======================================
        if self._contains_pronoun(question):

            topic = ChatManager.get_current_topic()

            if not topic:

                topic = self.topic_extractor.extract(
                    history_messages
                )

                if topic:

                    ChatManager.set_current_topic(
                        topic
                    )

            if topic:

                return self._replace_pronouns(
                    question,
                    topic
                )

            return question

        # ======================================
        # Handle commands like:
        # "continue", "tell me more", "more"
        # ======================================
        topic = ChatManager.get_current_topic()

        if topic:

            followup = self._handle_followup_command(
                question,
                topic
            )

            if followup != question:

                return followup

        # ======================================
        # Only extract new topic when the question
        # is NOT a pronoun-based follow-up.
        # ======================================
        new_topic = self.topic_extractor.extract_new_topic(
            question
        )

        if new_topic:

            ChatManager.set_current_topic(
                new_topic
            )

        return question

    def _contains_pronoun(
        self,
        question
    ):

        words = question.split()

        for word in words:

            clean = (
                word
                .lower()
                .strip(".,?!:;\"'")
            )

            if clean in self.SUBJECT_PRONOUNS:

                return True

            if clean in self.POSSESSIVE_PRONOUNS:

                return True

        return False

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

            clean_question = (
                question
                .strip()
                .rstrip(".?!")
            )

            return f"{clean_question} about {topic}"

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

            if word and word[-1] in ".,?!:;" :

                punctuation = word[-1]

            clean = (
                word
                .lower()
                .strip(".,?!:;\"'")
            )

            if clean in self.SUBJECT_PRONOUNS and not replaced:

                new_words.append(
                    topic + punctuation
                )

                replaced = True

                continue

            if clean in self.POSSESSIVE_PRONOUNS and not replaced:

                new_words.append(
                    topic + "'s" + punctuation
                )

                replaced = True

                continue

            new_words.append(
                word
            )

        return " ".join(
            new_words
        )