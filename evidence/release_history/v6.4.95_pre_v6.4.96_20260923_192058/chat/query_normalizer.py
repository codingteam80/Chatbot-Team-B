import re
import unicodedata


class QueryNormalizer:
    """
    Generic query normalizer for retrieval.

    Responsibilities:
    - Lowercase text
    - Remove accents
    - Remove punctuation safely while preserving Unicode letters
    - Preserve technical rule numbers like 10.1
    - Remove common question prefixes
    - Expand common abbreviations
    - Normalize whitespace
    """

    PREFIX_PATTERNS = [
        r"^who\s+is\s+",
        r"^who\s+was\s+",
        r"^who\s+are\s+",
        r"^sino\s+si\s+",
        r"^sino\s+ang\s+",
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
        # Identity / profile paraphrases. Keep the subject, normalize only the
        # generic way users ask for a short identity/overview.
        (
            r"^give\s+(?:me\s+)?(?:a\s+)?(?:short\s+|brief\s+|concise\s+)?(?:profile|description)\s+of\s+(.+?)(?:\s+from\s+(?:the\s+)?(?:knowledge\s+base|company\s+knowledge|stored\s+material|stored\s+documents|documents))?[?!.]*$",
            r"\1 biography"
        ),
        (
            r"^ano\s+ang\s+(.+?)\s+sa\s+(?:maikling|simpleng)\s+paliwanag[?!.]*$",
            r"\1 description"
        ),
        (
            r"^which\s+(?:two\s+)?countries\s+were\s+parties\s+to\s+(.+?)[?!.]*$",
            r"countries that signed \1"
        ),
        (
            r"^who\s+were\s+(?:the\s+)?(?:two\s+)?(?:national\s+)?parties\s+to\s+(.+?)[?!.]*$",
            r"countries that signed \1"
        ),
        (
            r"^sino\s+ang\s+(?:isa\s+sa\s+mga\s+)?nagtatag\s+(?:ng\s+)?(nito|niyan|niyon|ito|iyan|iyon)[?!.]*$",
            r"founder \1"
        ),
        (
            r"^(?:briefly\s+)?describe\s+(.+?)(?:\s+(?:briefly|in\s+a\s+sentence))?[?!.]*$",
            r"\1 biography"
        ),
        (
            r"^how\s+do\s+(?:the\s+)?(?:documents|stored\s+documents|stored\s+material)\s+identify\s+(.+?)[?!.]*$",
            r"\1 biography"
        ),
        (
            r"^how\s+is\s+(.+?)\s+(?:identified|described)\s+in\s+(?:the\s+)?(?:stored\s+material|stored\s+documents|documents|knowledge\s+base)[?!.]*$",
            r"\1 biography"
        ),
        (
            r"^(?:what\s+is\s+)?(?:the\s+)?(?:yearly|annual)\s+(.+?)\s+(?:allotment|allowance|entitlement)(?:\s+in\s+(?:the\s+)?(?:policy|document))?[?!.]*$",
            r"annual \1 amount"
        ),

        (
            r"^sino\s+ang\s+(?:isa\s+sa\s+mga\s+)?nagtatag\s+ng\s+(.+?)[?!.]*$",
            r"founder \1"
        ),
        (
            r"^sino[-\s]*sino\s+ang\s+(?:mga\s+)?nagtatag\s+ng\s+(.+?)[?!.]*$",
            r"founders \1"
        ),

        # Eligibility / qualification
        (
            r"^sino\s+(?:ang\s+)?(?:pwede|puwede|pwedeng|puwedeng|maaaring)\s+(?:gumamit|makagamit)\s+(?:nitong|nito|ng|sa)?\s*(.+?)(?:\s+benefits?)?[?!.]*$",
            r"\1 eligibility"
        ),
        (
            r"^who\s+(?:can|may)\s+use\s+(.+?)[?!.]*$",
            r"\1 eligibility"
        ),
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
        (
            r"^who\s+qualifies\s+for\s+(.+?)[?!.]*$",
            r"\1 eligibility"
        ),
        (
            r"^which\s+(.+?)\s+qualif(?:y|ies)\s+for\s+(.+?)[?!.]*$",
            r"\2 eligibility \1"
        ),
        (
            r"^who\s+(?:is|are)\s+covered\s+by\s+(.+?)[?!.]*$",
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
            r"^(?:okay\s*[,;:]?\s*)?sino\s+(?:naman\s+)?(?:ang\s+)?(?:nag[- ]?aapprove|nag[- ]?approve|umaapprove|mag[- ]?approve|nag[- ]?aapruba|mag[- ]?apruba)(?:\s+(?:ng|sa)\s+(.+?))?[?!.]*$",
            r"\1 approval"
        ),
        (
            r"^kanino\s+(?:kailangang\s+)?(?:ipa[- ]?approve|ipa[- ]?apruba)\s+(.+?)[?!.]*$",
            r"\1 approval"
        ),
        (
            r"^who\s+(?:can\s+)?(?:approve|authorize)\s+(.+?)[?!.]*$",
            r"\1 approval"
        ),
        (
            r"^which\s+(.+?)\s+(?:can\s+)?(?:approve|authorize)\s+(.+?)[?!.]*$",
            r"\2 approval \1"
        ),
        (
            r"^who\s+(?:must|needs?\s+to|required\s+to)\s+(?:approve|authorize)\s+(.+?)[?!.]*$",
            r"\1 approval"
        ),
        (
            r"^whose\s+approval\s+is\s+required\s+for\s+(.+?)[?!.]*$",
            r"\1 approval"
        ),
        (
            r"^whose\s+approval\s+does\s+(.+?)\s+require[?!.]*$",
            r"\1 approval"
        ),
        (
            r"^which\s+(.+?)\s+must\s+(?:approve|authorize)\s+(.+?)[?!.]*$",
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

        """
        Remove accents from Latin characters while preserving
        meaningful marks in non-Latin scripts.

        Example:
        - José -> Jose
        - です -> です
        """

        output = []

        for original_character in text:

            character_name = ""

            try:

                character_name = unicodedata.name(
                    original_character
                )

            except ValueError:

                output.append(
                    original_character
                )

                continue

            if "LATIN" not in character_name:

                output.append(
                    original_character
                )

                continue

            decomposed = unicodedata.normalize(
                "NFKD",
                original_character
            )

            output.extend(
                character
                for character in decomposed
                if not unicodedata.combining(
                    character
                )
            )

        return "".join(
            output
        )

    @staticmethod
    def _has_explicit_compound_intent(query: str) -> bool:

        """Return True when a second explicit question clause is present.

        Canonical single-intent rewrites such as ``X eligibility Y`` must not
        reorder a compound question like ``What is X and who is eligible?``.
        Keeping both clauses intact also lets the downstream compound-facet
        retriever preserve the user's requested order.
        """

        if not query:
            return False

        interrogative = (
            r"(?:who|what|when|where|why|how|which|"
            r"sino|ano|anong|kailan|saan|bakit|paano|alin)"
        )
        imperative = (
            r"(?:explain|describe|summari[sz]e|define|list|enumerate|"
            r"give|show|provide|tell|clarify|elaborate|"
            r"ipaliwanag|ilarawan|ibuod|ilista|ibigay)"
        )

        return bool(
            re.search(
                rf"\b(?:and|at)\b\s+(?:also\s+|din\s+|rin\s+)?"
                rf"(?=(?:{interrogative}|{imperative})\b)",
                query,
                flags=re.IGNORECASE,
            )
            or re.search(
                rf"[?;]\s*(?:please\s+|paki\s*)?"
                rf"(?=(?:{interrogative}|{imperative})\b)",
                query,
                flags=re.IGNORECASE,
            )
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

        # Convert equivalent single-intent eligibility/entitlement questions
        # into one stable retrieval form. Explicit compound questions keep
        # their original clause structure so a later clause cannot consume
        # and reorder the first one.
        if not self._has_explicit_compound_intent(query):
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

        # Preserve high-value C logical operators as semantic words before
        # generic punctuation stripping.  This prevents questions about the
        # right operand of && / || from collapsing into a vague "operand of or"
        # query and improves both single-query and MultiQuery retrieval.
        query = query.replace("&&", " logical and operator ")
        query = query.replace("||", " logical or operator ")

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
            r"[^\w\s]",
            " ",
            query,
            flags=re.UNICODE
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