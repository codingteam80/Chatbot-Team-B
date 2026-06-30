import re


class QueryNormalizer:
    """
    ==========================================================
    Query Normalizer

    Converts different question patterns into a standardized
    retrieval query.

    This improves retrieval consistency without requiring
    any additional LLM calls.

    Example

    Who is Jose Rizal?
        -> Jose Rizal

    Tell me about Jose Rizal.
        -> Jose Rizal

    Explain ISO 9001.
        -> ISO 9001
    ==========================================================
    """

    # ======================================================
    # QUESTION PREFIXES
    # ======================================================

    PREFIX_PATTERNS = [

        r"^who\s+is\s+",
        r"^who\s+was\s+",

        r"^what\s+is\s+",
        r"^what\s+was\s+",

        r"^tell\s+me\s+about\s+",

        r"^what\s+can\s+you\s+tell\s+me\s+about\s+",

        r"^give\s+me\s+information\s+about\s+",

        r"^provide\s+information\s+about\s+",

        r"^explain\s+",

        r"^describe\s+",

        r"^can\s+you\s+explain\s+",

        r"^can\s+you\s+tell\s+me\s+about\s+",
    ]

    # ======================================================
    # NORMALIZE
    # ======================================================

    def normalize(
        self,
        question: str
    ):

        if not question:

            return question

        query = question.strip()

        # Remove ending punctuation
        query = query.rstrip("?.! ")

        # Lowercase only for matching
        lower_query = query.lower()

        # Remove known prefixes
        for pattern in self.PREFIX_PATTERNS:

            if re.match(pattern, lower_query):

                query = re.sub(
                    pattern,
                    "",
                    query,
                    flags=re.IGNORECASE
                )

                break

        # Remove duplicate spaces
        query = re.sub(
            r"\s+",
            " ",
            query
        )

        return query.strip()