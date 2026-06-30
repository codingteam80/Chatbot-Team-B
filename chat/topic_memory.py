class TopicMemory:
    """
    ==========================================================
    Topic Memory

    Stores the current conversation topic.

    This allows follow-up questions such as:

        Where was he born?

        What was his occupation?

        When did he die?

    to continue referring to the same topic until
    the user explicitly changes it.

    This class contains no AI logic.

    It simply stores and retrieves the active topic.
    ==========================================================
    """

    def __init__(self):

        self._topic = None

    def get(self):
        """
        Return the current topic.
        """

        return self._topic

    def set(
        self,
        topic
    ):
        """
        Save a new topic.

        Ignore empty values.
        """

        if not topic:
            return

        topic = topic.strip()

        if topic:

            self._topic = topic

    def clear(self):
        """
        Clear the current topic.
        """

        self._topic = None

    def has_topic(self):
        """
        Returns True if a topic exists.
        """

        return self._topic is not None