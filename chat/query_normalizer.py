import re
import unicodedata


class QueryNormalizer:
    """
    Generic query normalizer for retrieval.

    Responsibilities:
    - Lowercase text
    - Remove accents
    - Remove punctuation safely
    - Preserve technical rule numbers like 10.1
    - Remove common question prefixes
    - Expand common abbreviations
    - Normalize whitespace
    """

    PREFIX_PATTERNS = [
        r"^who\s+is\s+",
        r"^who\s+was\s+",
        r"^who\s+are\s+",
        r"^what\s+is\s+",
        r"^what\s+was\s+",
        r"^what\s+are\s+",
        r"^where\s+is\s+",
        r"^where\s+was\s+",
        r"^when\s+is\s+",
        r"^when\s+was\s+",
        r"^tell\s+me\s+about\s+",
        r"^what\s+can\s+you\s+tell\s+me\s+about\s+",
        r"^give\s+me\s+information\s+about\s+",
        r"^provide\s+information\s+about\s+",
        r"^can\s+you\s+explain\s+",
        r"^can\s+you\s+tell\s+me\s+about\s+",
        r"^please\s+explain\s+",
        r"^explain\s+",
        r"^describe\s+",
        r"^define\s+",
        r"^summarize\s+",
    ]

    ABBREVIATIONS = {
        "vl": "vacation leave",
        "sl": "sick leave",
        "el": "emergency leave",
        "pl": "paternity leave",
        "ml": "maternity leave",
        "ot": "overtime",
        "hr": "human resources",
        "coe": "certificate of employment",
        "loa": "leave of absence",
        "dept": "department",
        "yr": "year",
    }

    @staticmethod
    def _remove_accents(text: str) -> str:

        normalized = unicodedata.normalize(
            "NFKD",
            text
        )

        return "".join(
            char for char in normalized
            if not unicodedata.combining(char)
        )

    def normalize(
        self,
        question: str
    ):

        if not question:

            return question

        query = question.strip().lower()

        query = self._remove_accents(
            query
        )

        # Preserve decimal rule numbers:
        # rule 10.1 should stay rule 10.1
        query = re.sub(
            r"(?<=\d)\.(?=\d)",
            "__dot__",
            query
        )

        # Remove common question prefixes
        for pattern in self.PREFIX_PATTERNS:

            if re.match(
                pattern,
                query
            ):

                query = re.sub(
                    pattern,
                    "",
                    query,
                    flags=re.IGNORECASE
                )

                break

        # Replace punctuation/symbols with space,
        # but preserve underscores for __dot__ marker.
        query = re.sub(
            r"[^a-z0-9\s_]",
            " ",
            query
        )

        # Restore decimal dots
        query = query.replace(
            "__dot__",
            "."
        )

        # Normalize spaces
        query = re.sub(
            r"\s+",
            " ",
            query
        ).strip()

        if not query:

            return query

        expanded_words = []

        for word in query.split():

            replacement = self.ABBREVIATIONS.get(
                word,
                word
            )

            expanded_words.extend(
                replacement.split()
            )

        query = " ".join(
            expanded_words
        )

        query = re.sub(
            r"\s+",
            " ",
            query
        ).strip()

        return query