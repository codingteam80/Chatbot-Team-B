class QueryEnricher:
    """
    ==========================================================
    Query Enricher

    Expands simple topic-only queries into richer retrieval
    queries to improve recall.

    Examples

    Jose Rizal
        ->
    Explain Jose Rizal.
    Include overview, background,
    important facts and key information.

    ISO 9001
        ->
    Explain ISO 9001.
    Include overview, purpose,
    requirements, benefits,
    important details.

    Employee Handbook
        ->
    Explain Employee Handbook.
    Include overview,
    policies,
    guidelines,
    important rules.

    Questions such as

    Where was Jose Rizal born?

    Who created ISO 9001?

    Compare ISO 9001 and ISO 14001

    are NOT modified.
    ==========================================================
    """

    QUESTION_WORDS = {

        "who",
        "what",
        "where",
        "when",
        "why",
        "how",
        "which",
        "whose"
    }

    ACTION_WORDS = {

        "compare",
        "list",
        "show",
        "find",
        "search",
        "locate",
        "give",
        "provide",
        "describe",
        "explain",
        "tell",
        "define",
        "summarize"
    }

    MAX_TOPIC_WORDS = 6

    def enrich(
        self,
        query: str
    ):

        if not query:

            return query

        clean = (
            query
            .strip()
            .rstrip("?.!")
        )

        words = clean.split()

        if not words:

            return query

        first = words[0].lower()

        # Already a normal question
        if first in self.QUESTION_WORDS:

            return query

        # Already an action query
        if first in self.ACTION_WORDS:

            return query

        # Long query
        if len(words) > self.MAX_TOPIC_WORDS:

            return query

        return f"{clean} overview"