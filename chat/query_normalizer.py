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

    # Canonical rewrites for equivalent retrieval intents.
    # These run before generic prefix removal so important words
    # such as eligibility and entitlement are preserved.
    CANONICAL_INTENT_PATTERNS = [
        # Eligibility / qualification
        (
            r"^who\s+(?:is|are)\s+(?:eligible|qualified)\s+for\s+(.+?)[?!.]*$",
            r"\1 eligibility"
        ),
        (
            r"^which\s+(.+?)\s+(?:is|are)\s+(?:eligible|qualified)\s+for\s+(.+?)[?!.]*$",
            r"\2 eligibility \1"
        ),
        (
            r"^(.+?)\s+(?:is|are)\s+(?:eligible|qualified)\s+for\s+(.+?)[?!.]*$",
            r"\2 eligibility \1"
        ),
        (
            r"^(?:eligible|qualified)\s+for\s+(.+?)[?!.]*$",
            r"\1 eligibility"
        ),
        (
            r"^(?:eligibility|qualification)\s+for\s+(.+?)[?!.]*$",
            r"\1 eligibility"
        ),

        # Entitlement
        (
            r"^who\s+(?:is|are)\s+entitled\s+to\s+(.+?)[?!.]*$",
            r"\1 entitlement"
        ),
        (
            r"^which\s+(.+?)\s+(?:is|are)\s+entitled\s+to\s+(.+?)[?!.]*$",
            r"\2 entitlement \1"
        ),
        (
            r"^entitled\s+to\s+(.+?)[?!.]*$",
            r"\1 entitlement"
        ),
        (
            r"^entitlement\s+to\s+(.+?)[?!.]*$",
            r"\1 entitlement"
        ),

        # Approval
        (
            r"^who\s+(?:can\s+)?(?:approve|authorize)\s+(.+?)[?!.]*$",
            r"\1 approval"
        ),
        (
            r"^which\s+(.+?)\s+(?:can\s+)?(?:approve|authorize)\s+(.+?)[?!.]*$",
            r"\2 approval \1"
        ),

        # Authorization / permission
        (
            r"^who\s+(?:can|may)\s+(.+?)[?!.]*$",
            r"\1 authorization"
        ),
        (
            r"^which\s+(.+?)\s+(?:can|may)\s+(.+?)[?!.]*$",
            r"\2 authorization \1"
        ),
        (
            r"^who\s+(?:is|are)\s+(?:authorized|allowed|permitted)\s+to\s+(.+?)[?!.]*$",
            r"\1 authorization"
        ),

        # Responsibility / ownership
        (
            r"^who\s+(?:is|are)\s+responsible\s+for\s+(.+?)[?!.]*$",
            r"\1 responsibility"
        ),
        (
            r"^which\s+(.+?)\s+(?:is|are)\s+responsible\s+for\s+(.+?)[?!.]*$",
            r"\2 responsibility \1"
        ),

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

        # Convert equivalent eligibility/entitlement questions
        # into one stable retrieval form.
        for pattern, replacement in (
            self.CANONICAL_INTENT_PATTERNS
        ):

            if re.match(
                pattern,
                query,
                flags=re.IGNORECASE
            ):

                query = re.sub(
                    pattern,
                    replacement,
                    query,
                    count=1,
                    flags=re.IGNORECASE
                )

                break

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