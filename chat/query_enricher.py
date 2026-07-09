import re


class QueryEnricher:
    """
    Rule-based query enricher for DocuBot.

    Purpose:
    - Improve retrieval quality for technical/company documents.
    - Improve topic overview questions.
    - Improve consistency between similar questions.
    - No LLM call is used here.
    - This does NOT generate the answer.
    - This only improves the search query used by retrieval.
    """

    def __init__(self):

        # Questions that usually ask for a topic overview.
        self.overview_patterns = [
            r"\bwhat can you tell me about\b",
            r"\btell me about\b",
            r"\bexplain\b",
            r"\bdescribe\b",
            r"\bgive me an overview\b",
            r"\boverview of\b",
            r"\bsummary of\b",
            r"\bwhat is\b",
            r"\bano ang\b",
            r"\btungkol sa\b",
            r"\bipaliwanag\b",
            r"\bpakiexplain\b",
            r"\bpaki explain\b",
            r"\bano ang tungkol sa\b",
        ]

        # Questions that usually ask about a person/entity identity.
        self.identity_patterns = [
            r"\bwho is\b",
            r"\bwho was\b",
            r"\bsino si\b",
            r"\bsino ang\b",
            r"\bkilala si\b",
            r"\bkilala ang\b",
        ]

        # Questions asking who qualifies, is entitled, or is allowed.
        # These are policy/eligibility requests, not person-identity
        # questions even when they begin with "who is".
        self.eligibility_patterns = [
            # Eligibility / qualification
            r"\b(?:eligible|eligibility|qualified|qualification)\b",

            # Entitlement
            r"\b(?:entitled|entitlement)\b",

            # Authorization / permission
            r"\b(?:authorized|authorization|allowed|permission|permitted)\b",
            r"\bwho\s+(?:can|may)\b",
            r"\bwhich\s+.+?\s+(?:can|may)\b",

            # Responsibility / ownership
            r"\b(?:responsible|responsibility|owner|ownership)\b",

            # Approval
            r"\b(?:approve|approval|approver|authorize)\b",
        ]

        # Questions asking for multiple answers / lists.
        self.list_patterns = [
            r"\bwho are\b",
            r"\bwhat are\b",
            r"\blist\b",
            r"\benumerate\b",
            r"\bname the\b",
            r"\bgive me the list\b",
            r"\bsino sino\b",
            r"\bano ano\b",
            r"\bilista\b",
        ]

        # Questions asking about relationships / people connected to a person.
        self.relationship_patterns = [
            r"\brelationship with\b",
            r"\brelationships with\b",
            r"\bromantic relationship\b",
            r"\bromantic relationships\b",
            r"\blove interest\b",
            r"\blove interests\b",
            r"\bgirlfriend\b",
            r"\bgirlfriends\b",
            r"\bwomen\b",
            r"\bladies\b",
            r"\bassociated with\b",
            r"\bconnected to\b",
        ]

        # Generic words that should be removed when extracting the topic.
        self.stop_phrases = [
            "what can you tell me about",
            "tell me about",
            "give me an overview of",
            "overview of",
            "summary of",
            "explain",
            "describe",
            "what is",
            "who is",
            "who was",
            "ano ang tungkol kay",
            "ano ang tungkol sa",
            "ano ang",
            "tungkol kay",
            "tungkol sa",
            "ipaliwanag",
            "pakiexplain",
            "paki explain",
            "sino si",
            "sino ang",
            "please",
            "pls",
            "po",
            "ba",
        ]

        # General overview terms.
        self.general_overview_terms = [
            "definition",
            "overview",
            "description",
            "summary",
            "background",
            "purpose",
            "scope",
            "details",
            "important information",
        ]

        # Technical document terms useful for IT/manual/SOP retrieval.
        self.technical_overview_terms = [
            "definition",
            "purpose",
            "scope",
            "overview",
            "description",
            "policy",
            "standard",
            "procedure",
            "process",
            "requirements",
            "guidelines",
            "responsibility",
            "roles",
            "steps",
            "configuration",
            "setup",
            "operation",
            "troubleshooting",
            "exception",
            "notes",
        ]

        # Terms useful when user asks about procedure/process.
        self.process_terms = [
            "procedure",
            "process",
            "steps",
            "workflow",
            "sequence",
            "requirements",
            "responsibility",
            "approval",
            "verification",
        ]

        # Terms useful when user asks about policy/standard.
        self.policy_terms = [
            "policy",
            "standard",
            "rules",
            "requirements",
            "compliance",
            "scope",
            "exception",
            "responsibility",
        ]

        # Terms useful when user asks about configuration/setup.
        self.configuration_terms = [
            "configuration",
            "setup",
            "installation",
            "settings",
            "parameters",
            "requirements",
            "troubleshooting",
        ]

        # Technical keywords that help determine if the question is about IT/SOP/manual content.
        self.technical_keywords = [
            "policy",
            "standard",
            "procedure",
            "process",
            "workflow",
            "configuration",
            "setup",
            "install",
            "installation",
            "settings",
            "system",
            "network",
            "server",
            "database",
            "backup",
            "restore",
            "security",
            "access",
            "account",
            "password",
            "software",
            "hardware",
            "troubleshooting",
            "operation",
            "manual",
            "guideline",
            "requirement",
        ]

    def enrich(
        self,
        question: str,
        intent_question: str | None = None
    ) -> str:
        """
        Main public method.

        Input:
            question: user question after normalization/resolution
            intent_question: original user question before normalization

        Output:
            enriched query string for retrieval
        """

        if not question:
            return ""

        original_question = question.strip()
        lowered_question = original_question.lower()

        intent_text = (
            intent_question
            if intent_question
            else original_question
        ).strip()

        lowered_intent = intent_text.lower()

        topic = self._extract_topic(
            original_question
        )

        relationship_topic = (
            self._extract_relationship_topic(
                intent_text
            )
            or self._extract_relationship_topic(
                original_question
            )
        )

        if relationship_topic:

            topic = relationship_topic

        enrichment_terms = []

        is_eligibility_question = (
            self._is_eligibility_question(
                lowered_intent
                + " "
                + lowered_question
            )
        )

        is_identity_question = (
            self._is_identity_question(
                lowered_intent
            )
            and not is_eligibility_question
        )

        is_overview_question = self._is_overview_question(
            lowered_intent
        )

        is_technical_question = self._is_technical_question(
            lowered_question
            + " "
            + lowered_intent
        )

        is_list_question = self._is_list_question(
            lowered_intent
        )

        is_relationship_question = self._is_relationship_question(
            lowered_intent
            + " "
            + lowered_question
        )

        looks_like_person_name = self._looks_like_person_name(
            intent_text
        )

        # ======================================
        # Eligibility / Entitlement Questions
        # ======================================
        if is_eligibility_question:

            eligibility_subject = (
                self._extract_eligibility_subject(
                    original_question,
                    intent_text
                )
            )

            enrichment_terms.extend(
                self._build_eligibility_terms(
                    eligibility_subject
                    or topic
                    or original_question,
                    intent_text=(
                        lowered_intent
                        + " "
                        + lowered_question
                    )
                )
            )

        # ======================================
        # Relationship / Multi-Answer List Questions
        # ======================================
        elif (
            is_relationship_question
            and topic
        ):

            enrichment_terms.extend(
                self._build_relationship_terms(
                    topic
                )
            )

        # ======================================
        # Identity / Person / Entity Questions
        # ======================================
        # Important:
        # Do NOT add broad generic terms alone like:
        # background, overview, summary, description.
        #
        # Those generic words can accidentally match unrelated
        # technical documents.
        #
        # Instead, keep every enrichment phrase anchored to the topic.
        elif (
            is_identity_question
            or (
                is_overview_question
                and looks_like_person_name
            )
        ):

            if topic:

                enrichment_terms.extend(
                    self._build_identity_terms(
                        topic
                    )
                )

            else:

                enrichment_terms.append(
                    original_question
                )

        # ======================================
        # General Overview Questions
        # ======================================
        elif is_overview_question:

            # If the topic is technical/company-related,
            # generic technical overview terms are useful.
            if is_technical_question:

                # Keep technical overview terms anchored
                # to the actual topic.
                #
                # Avoid adding bare words such as:
                # requirements, rules, steps, procedure,
                # configuration, and troubleshooting.
                #
                # Bare terms can incorrectly activate
                # list/completeness mode.
                if topic:

                    enrichment_terms.extend(
                        self._build_topic_overview_terms(
                            topic
                        )
                    )

                else:

                    enrichment_terms.append(
                        original_question
                    )

            # If the topic is NOT technical, avoid generic terms alone.
            # Keep the query anchored to the topic name.
            else:

                if topic:

                    enrichment_terms.extend(
                        self._build_topic_overview_terms(
                            topic
                        )
                    )

                else:

                    enrichment_terms.extend(
                        self.general_overview_terms
                    )

        # ======================================
        # Procedure / Process Questions
        # ======================================
        if self._has_any(
            lowered_question,
            [
                "procedure",
                "process",
                "steps",
                "workflow",
                "how to",
                "paano",
            ]
        ):

            enrichment_terms.extend(
                self.process_terms
            )

        # ======================================
        # Policy / Standard Questions
        # ======================================
        if self._has_any(
            lowered_question,
            [
                "policy",
                "standard",
                "rule",
                "rules",
                "compliance",
            ]
        ):

            enrichment_terms.extend(
                self.policy_terms
            )

        # ======================================
        # Configuration / Setup Questions
        # ======================================
        if self._has_any(
            lowered_question,
            [
                "configure",
                "configuration",
                "setup",
                "install",
                "installation",
                "settings",
            ]
        ):

            enrichment_terms.extend(
                self.configuration_terms
            )

        # ======================================
        # Yes/No Claim Retrieval Compatibility
        # ======================================
        enrichment_terms.extend(
            self._build_boolean_claim_terms(
                intent_text
            )
        )

        enrichment_terms = self._deduplicate(
            enrichment_terms
        )

        # If no enrichment is needed, return original question.
        if not enrichment_terms:

            return original_question

        enriched_query_parts = [
            original_question,
        ]

        if topic and topic.lower() != original_question.lower():

            enriched_query_parts.append(
                topic
            )

        enriched_query_parts.extend(
            enrichment_terms
        )

        return " ".join(
            self._deduplicate(
                enriched_query_parts
            )
        )

    def _simple_past_form(
        self,
        verb: str
    ) -> str:

        """
        Return a retrieval-friendly past form.

        Exact grammar is less important than adding a useful
        semantic and lexical search variant.
        """

        irregular = {
            "be": "was",
            "begin": "began",
            "build": "built",
            "buy": "bought",
            "cause": "caused",
            "come": "came",
            "create": "created",
            "do": "did",
            "find": "found",
            "found": "founded",
            "get": "got",
            "give": "gave",
            "go": "went",
            "have": "had",
            "kill": "killed",
            "lead": "led",
            "make": "made",
            "run": "ran",
            "send": "sent",
            "take": "took",
            "write": "wrote",
        }

        clean = (
            verb or ""
        ).strip().lower()

        if not clean:

            return ""

        if clean in irregular:

            return irregular[clean]

        if clean.endswith("e"):

            return clean + "d"

        if clean.endswith("y") and len(clean) > 1:

            return clean[:-1] + "ied"

        return clean + "ed"

    def _join_repeated_name_tokens(
        self,
        text: str
    ) -> str:

        """
        Add compatibility for names or identifiers that may be
        indexed with or without a space.

        Example:
            token token -> tokentoken
        """

        if not text:

            return ""

        tokens = text.split()
        output = []
        index = 0

        while index < len(tokens):

            if (
                index + 1 < len(tokens)
                and tokens[index].lower()
                == tokens[index + 1].lower()
            ):

                output.append(
                    tokens[index]
                    + tokens[index + 1]
                )

                index += 2

                continue

            output.append(
                tokens[index]
            )

            index += 1

        return " ".join(
            output
        )

    def _build_boolean_claim_terms(
        self,
        question: str
    ) -> list[str]:

        """
        Expand yes/no claims into retrieval-friendly forms.

        This helps semantically equivalent queries retrieve the
        same evidence as direct WH questions.

        Example structure:
            Did SUBJECT VERB OBJECT?
            SUBJECT VERB OBJECT
            SUBJECT PAST_VERB OBJECT
            OBJECT PAST_VERB by SUBJECT
            who PAST_VERB OBJECT
        """

        if not question:

            return []

        clean = re.sub(
            r"[?!.]+$",
            "",
            question.strip()
        )

        match = re.match(
            r"^(did|does|do|is|are|was|were|has|have|had|"
            r"can|could|will|would|should|must)\s+(.+)$",
            clean,
            flags=re.IGNORECASE
        )

        if not match:

            return []

        auxiliary = match.group(1).lower()
        claim = match.group(2).strip()

        if not claim:

            return []

        terms = [
            claim,
            self._join_repeated_name_tokens(
                claim
            ),
            f"evidence {claim}",
            f"statement {claim}",
        ]

        # For action claims, create active/passive/WH variants.
        action_verbs = {
            "access",
            "add",
            "allow",
            "approve",
            "assign",
            "authorize",
            "build",
            "cause",
            "configure",
            "create",
            "delete",
            "deploy",
            "design",
            "develop",
            "establish",
            "execute",
            "found",
            "implement",
            "install",
            "invent",
            "kill",
            "lead",
            "make",
            "manage",
            "modify",
            "own",
            "perform",
            "publish",
            "release",
            "require",
            "run",
            "start",
            "stop",
            "support",
            "update",
            "use",
            "write",
        }

        claim_tokens = claim.split()
        verb_index = None

        for index, token in enumerate(
            claim_tokens
        ):

            normalized_token = re.sub(
                r"[^A-Za-z'-]",
                "",
                token
            ).lower()

            if normalized_token in action_verbs:

                verb_index = index
                break

        if (
            auxiliary in {
                "did",
                "does",
                "do",
            }
            and verb_index is not None
            and verb_index > 0
            and verb_index < len(claim_tokens) - 1
        ):

            subject = " ".join(
                claim_tokens[:verb_index]
            )

            verb = re.sub(
                r"[^A-Za-z'-]",
                "",
                claim_tokens[verb_index]
            )

            object_text = " ".join(
                claim_tokens[verb_index + 1:]
            )

            past_verb = self._simple_past_form(
                verb
            )

            compact_subject = (
                self._join_repeated_name_tokens(
                    subject
                )
            )

            terms.extend(
                [
                    f"{subject} {past_verb} {object_text}",
                    f"{compact_subject} {past_verb} {object_text}",
                    f"{object_text} {past_verb} by {subject}",
                    f"{object_text} {past_verb} by {compact_subject}",
                    f"who {past_verb} {object_text}",
                ]
            )

        return self._deduplicate(
            [
                term
                for term in terms
                if term
            ]
        )


    def _build_eligibility_terms(
        self,
        subject: str,
        intent_text: str = ""
    ) -> list[str]:
        """
        Build generic, topic-anchored retrieval terms.

        This works across company policies, technical manuals,
        access rules, approvals, responsibilities, benefits,
        systems, procedures, and other document types.
        """

        clean_subject = re.sub(
            r"\s+",
            " ",
            subject or ""
        ).strip()

        if not clean_subject:

            return []

        intent_type = self._detect_eligibility_intent(
            intent_text
            + " "
            + clean_subject
        )

        terms = [
            clean_subject
        ]

        if intent_type == "authorization":

            terms.extend(
                [
                    f"{clean_subject} authorization",
                    f"authorized to {clean_subject}",
                    f"permission to {clean_subject}",
                    f"permitted roles {clean_subject}",
                    f"{clean_subject} access requirements",
                    f"{clean_subject} allowed roles",
                ]
            )

        elif intent_type == "responsibility":

            terms.extend(
                [
                    f"{clean_subject} responsibility",
                    f"responsible for {clean_subject}",
                    f"{clean_subject} owner",
                    f"{clean_subject} assigned role",
                    f"{clean_subject} duties",
                ]
            )

        elif intent_type == "approval":

            terms.extend(
                [
                    f"{clean_subject} approval",
                    f"approve {clean_subject}",
                    f"{clean_subject} approver",
                    f"authorized approver {clean_subject}",
                    f"{clean_subject} approval authority",
                ]
            )

        elif intent_type == "entitlement":

            terms.extend(
                [
                    f"{clean_subject} entitlement",
                    f"entitled to {clean_subject}",
                    f"{clean_subject} eligibility",
                    f"{clean_subject} conditions",
                    f"{clean_subject} requirements",
                ]
            )

        else:

            terms.extend(
                [
                    f"{clean_subject} eligibility",
                    f"eligible for {clean_subject}",
                    f"{clean_subject} qualification",
                    f"{clean_subject} requirements",
                    f"{clean_subject} conditions",
                    f"{clean_subject} criteria",
                ]
            )

        return self._deduplicate(
            terms
        )

    def _detect_eligibility_intent(
        self,
        text: str
    ) -> str:
        """
        Identify the generic intent family without relying on
        a specific document topic.
        """

        clean = (
            text
            or ""
        ).lower()

        if re.search(
            r"\b(approve|approval|approver)\b",
            clean
        ):

            return "approval"

        if re.search(
            r"\b(responsible|responsibility|owner|ownership)\b",
            clean
        ):

            return "responsibility"

        if re.search(
            r"\b(authorized|authorization|allowed|permission|permitted)\b",
            clean
        ) or re.search(
            r"\b(?:who|which\s+.+?)\s+(?:can|may)\b",
            clean
        ):

            return "authorization"

        if re.search(
            r"\b(entitled|entitlement)\b",
            clean
        ):

            return "entitlement"

        return "eligibility"

    def _extract_eligibility_subject(
        self,
        normalized_question: str,
        intent_question: str
    ) -> str:
        """
        Extract the subject of a generic role/policy intent.

        Examples:
            leave eligibility -> leave
            deploy production authorization -> deploy production
            incident response responsibility -> incident response
            purchase request approval -> purchase request
        """

        candidates = [
            normalized_question,
            intent_question,
        ]

        patterns = [
            # Canonical normalized forms
            r"^(.+?)\s+(?:eligibility|entitlement|authorization|responsibility|approval)$",
            r"^(.+?)\s+(?:eligibility|entitlement|authorization|responsibility|approval)\s+.+$",

            # Natural eligibility / entitlement wording
            r"^(?:eligibility|qualification)\s+for\s+(.+)$",
            r"^(?:eligible|qualified)\s+for\s+(.+)$",
            r"^entitlement\s+to\s+(.+)$",
            r"^entitled\s+to\s+(.+)$",
            r"^who\s+(?:is|are)\s+(?:eligible|qualified)\s+for\s+(.+)$",
            r"^who\s+(?:is|are)\s+entitled\s+to\s+(.+)$",
            r"^which\s+.+?\s+(?:is|are)\s+(?:eligible|qualified)\s+for\s+(.+)$",

            # Authorization / permission
            r"^who\s+(?:can|may)\s+(.+)$",
            r"^which\s+.+?\s+(?:can|may)\s+(.+)$",
            r"^who\s+(?:is|are)\s+(?:authorized|allowed|permitted)\s+to\s+(.+)$",

            # Responsibility / approval
            r"^who\s+(?:is|are)\s+responsible\s+for\s+(.+)$",
            r"^which\s+.+?\s+(?:is|are)\s+responsible\s+for\s+(.+)$",
            r"^who\s+(?:can\s+)?(?:approve|authorize)\s+(.+)$",
            r"^which\s+.+?\s+(?:can\s+)?(?:approve|authorize)\s+(.+)$",
        ]

        for candidate in candidates:

            clean_candidate = (
                candidate
                .strip()
                .rstrip("?!.")
            )

            for pattern in patterns:

                match = re.match(
                    pattern,
                    clean_candidate,
                    flags=re.IGNORECASE
                )

                if not match:

                    continue

                subject = match.group(1).strip()

                subject = re.sub(
                    r"^(?:the|a|an)\s+",
                    "",
                    subject,
                    flags=re.IGNORECASE
                ).strip()

                return subject

        return ""

    def _build_identity_terms(
        self,
        topic: str
    ) -> list[str]:
        """
        Build identity/person enrichment terms.

        Every phrase is anchored to the topic name to avoid
        matching unrelated generic documents.
        """

        clean_topic = topic.strip()

        if not clean_topic:

            return []

        return [
            clean_topic,
            f"{clean_topic} biography",
            f"{clean_topic} life",
            f"{clean_topic} known for",
            f"{clean_topic} background",
            f"{clean_topic} important facts",
        ]

    def _looks_like_person_name(
        self,
        text: str
    ) -> bool:
        """
        Detect whether the extracted topic likely represents
        one person's name.

        This avoids classifying document titles, standards,
        policies, events, systems, organizations, and multiple
        entities as person names.
        """

        if not text:

            return False

        topic = self._extract_topic(
            text
        )

        if not topic:

            return False

        topic = re.sub(
            r"\s+",
            " ",
            topic
        ).strip()

        words = topic.split()

        # Most personal names contain around 2 to 5 words.
        if len(words) < 2:

            return False

        if len(words) > 5:

            return False

        normalized_topic = re.sub(
            r"[^a-zA-Z0-9\s'-]",
            " ",
            topic
        ).lower()

        normalized_topic = re.sub(
            r"\s+",
            " ",
            normalized_topic
        ).strip()

        # Numbers normally indicate a version, date,
        # rule, standard, or technical identifier.
        if re.search(
            r"\d",
            normalized_topic
        ):

            return False

        # Multiple joined entities should not be treated
        # as one person's name.
        if re.search(
            r"\b(and|or|versus|vs)\b",
            normalized_topic
        ):

            return False

        non_person_terms = {
            # Documents and governance
            "policy",
            "procedure",
            "standard",
            "standards",
            "manual",
            "guideline",
            "guidelines",
            "requirement",
            "requirements",
            "regulation",
            "regulations",
            "document",
            "report",
            "agreement",
            "treaty",
            "law",
            "act",
            "code",
            "rule",
            "rules",

            # Historical topics and events
            "history",
            "war",
            "revolution",
            "occupation",
            "battle",
            "movement",
            "empire",
            "government",

            # Technical topics
            "system",
            "software",
            "hardware",
            "network",
            "server",
            "database",
            "application",
            "platform",
            "framework",
            "protocol",
            "configuration",
            "installation",
            "setup",
            "security",
            "backup",
            "process",
            "workflow",
            "architecture",
            "project",
            "program",
            "model",
            "method",
            "api",

            # Organizations and groups
            "company",
            "organization",
            "organisation",
            "department",
            "division",
            "committee",
            "council",
            "university",
            "school",
            "institute",
            "association",
            "consortium",
            "bank",

            # Common company document topics
            "leave",
            "overtime",
            "benefit",
            "benefits",
            "misra",
        }

        topic_tokens = set(
            normalized_topic.split()
        )

        if topic_tokens.intersection(
            non_person_terms
        ):

            return False

        # Titles beginning with an article are commonly
        # concepts, documents, events, or organizations.
        if normalized_topic.startswith(
            (
                "the ",
                "a ",
                "an ",
            )
        ):

            return False

        # Name particles may correctly appear in lowercase.
        name_particles = {
            "de",
            "del",
            "dela",
            "la",
            "las",
            "los",
            "van",
            "von",
            "da",
            "di",
            "dos",
            "das",
            "bin",
            "al",
            "y",
        }

        significant_words = [
            word
            for word in words
            if word.lower() not in name_particles
        ]

        if len(significant_words) < 2:

            return False

        capitalized_words = [
            word
            for word in significant_words
            if word[:1].isupper()
        ]

        return len(capitalized_words) >= 2

    def _build_topic_overview_terms(
        self,
        topic: str
    ) -> list[str]:
        """
        Build overview enrichment terms that stay anchored
        to the actual topic.

        This prevents generic words like:
        definition, overview, background, purpose

        from matching unrelated documents.
        """

        clean_topic = topic.strip()

        if not clean_topic:
            return []

        return [
            clean_topic,
            f"{clean_topic} definition",
            f"{clean_topic} overview",
            f"{clean_topic} description",
            f"{clean_topic} summary",
            f"{clean_topic} background",
            f"{clean_topic} important information",
            f"{clean_topic} key facts",
        ]

    def _build_relationship_terms(
        self,
        topic: str
    ) -> list[str]:
        """
        Build relationship/list enrichment terms.

        Used for questions like:
            Who are the ladies that had relationship with Jose Rizal?
            List the women linked to Jose Rizal.
        """

        clean_topic = topic.strip()

        if not clean_topic:

            return []

        return [
            clean_topic,
            f"{clean_topic} relationship",
            f"{clean_topic} relationships",
            f"{clean_topic} romantic relationship",
            f"{clean_topic} romantic relationships",
            f"{clean_topic} personal relationships",
            f"{clean_topic} love interests",
            f"{clean_topic} romance",
            f"{clean_topic} sweetheart",
            f"{clean_topic} courtship",
            f"{clean_topic} wife",
            f"{clean_topic} marriage",
            f"{clean_topic} women",
            f"{clean_topic} ladies",
            f"{clean_topic} girlfriends",
            f"{clean_topic} personal life",
            f"{clean_topic} associated women",
        ]

    def _extract_relationship_topic(
        self,
        question: str
    ) -> str:
        """
        Extract topic from relationship/list questions.

        Example:
            Who are the ladies that had relationship with Jose Rizal?
            -> Jose Rizal
        """

        if not question:

            return ""

        text = question.strip()

        text = re.sub(
            r"[?!.]+$",
            "",
            text
        ).strip()

        patterns = [
            r"relationship\s+with\s+(.+)$",
            r"relationships\s+with\s+(.+)$",
            r"romantic\s+relationship\s+with\s+(.+)$",
            r"romantic\s+relationships\s+with\s+(.+)$",
            r"love\s+interests?\s+of\s+(.+)$",
            r"girlfriends?\s+of\s+(.+)$",
            r"women\s+(?:linked|connected|associated)\s+(?:to|with)\s+(.+)$",
            r"ladies\s+(?:linked|connected|associated)\s+(?:to|with)\s+(.+)$",
            r"associated\s+with\s+(.+)$",
            r"connected\s+to\s+(.+)$",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            if match:

                topic = match.group(1).strip()

                topic = re.sub(
                    r"^(with|to|of|about)\s+",
                    "",
                    topic,
                    flags=re.IGNORECASE
                ).strip()

                return topic

        return ""

    def _is_list_question(
        self,
        lowered_question: str
    ) -> bool:
        """
        Detect questions asking for multiple items.
        """

        for pattern in self.list_patterns:

            if re.search(
                pattern,
                lowered_question
            ):

                return True

        return False

    def _is_relationship_question(
        self,
        lowered_question: str
    ) -> bool:
        """
        Detect relationship / connected people questions.
        """

        for pattern in self.relationship_patterns:

            if re.search(
                pattern,
                lowered_question
            ):

                return True

        return False

    def _is_overview_question(
        self,
        lowered_question: str
    ) -> bool:
        """
        Detect broad topic overview questions.
        """

        for pattern in self.overview_patterns:

            if re.search(
                pattern,
                lowered_question
            ):

                return True

        return False

    def _is_eligibility_question(
        self,
        lowered_question: str
    ) -> bool:
        """
        Detect policy questions asking who qualifies or is entitled.

        This prevents "who is eligible..." from being treated as
        a biography/person identity query.
        """

        for pattern in self.eligibility_patterns:

            if re.search(
                pattern,
                lowered_question
            ):

                return True

        return False

    def _is_identity_question(
        self,
        lowered_question: str
    ) -> bool:
        """
        Detect questions asking who/what a person or entity is.
        """

        for pattern in self.identity_patterns:

            if re.search(
                pattern,
                lowered_question
            ):

                return True

        return False

    def _is_technical_question(
        self,
        lowered_question: str
    ) -> bool:
        """
        Detect if the question likely belongs to technical/company docs.
        """

        return self._has_any(
            lowered_question,
            self.technical_keywords
        )

    def _extract_topic(
        self,
        question: str
    ) -> str:
        """
        Remove generic overview phrases and keep the real topic.

        Example:
            "What can you tell me about password policy?"
            -> "password policy"
        """

        topic = question.strip()

        topic = re.sub(
            r"[?!.]+$",
            "",
            topic
        ).strip()

        lowered_topic = topic.lower()

        for phrase in self.stop_phrases:

            # Remove the phrase only when it appears
            # as a complete word or complete phrase
            # at the beginning of the question.
            pattern = (
                r"^"
                + re.escape(phrase)
                + r"(?:\s+|$)"
            )

            if re.search(
                pattern,
                lowered_topic,
                flags=re.IGNORECASE
            ):

                topic = re.sub(
                    pattern,
                    "",
                    topic,
                    count=1,
                    flags=re.IGNORECASE
                ).strip()

                break

        topic = re.sub(
            r"^(about|of|sa|ng|kay|si|ang)\s+",
            "",
            topic,
            flags=re.IGNORECASE
        ).strip()

        return topic

    def _has_any(
        self,
        text: str,
        keywords: list[str]
    ) -> bool:
        """
        Returns True if any keyword exists in the text.
        """

        return any(
            keyword in text
            for keyword in keywords
        )

    def _deduplicate(
        self,
        items: list[str]
    ) -> list[str]:
        """
        Removes duplicates while preserving order.
        """

        seen = set()
        output = []

        for item in items:

            clean_item = item.strip()

            if not clean_item:

                continue

            key = clean_item.lower()

            if key not in seen:

                seen.add(
                    key
                )

                output.append(
                    clean_item
                )

        return output