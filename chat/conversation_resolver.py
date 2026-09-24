import re

from chat.topic_extractor import TopicExtractor
from chat.chat_manager import ChatManager
from utils.structured_reference import extract_structured_reference


class ConversationResolver:
    """
    Resolves follow-up questions using the current conversation topic.

    Purpose:
    - Resolve pronouns like he, him, his, she, her.
    - Resolve object pronouns like it, its, this, that.
    - Resolve short follow-up questions using current topic.
    - Avoid treating generic follow-ups as new topics.
    """

    PERSON_PRONOUNS = {
        "he",
        "him",
        "she",
        "her",
        "siya",
    }

    OBJECT_PRONOUNS = {
        "it",
        "this",
        "that",
        "ito",
        "iyan",
        "iyon",
    }

    PLURAL_PRONOUNS = {
        "they",
        "them",
        "these",
        "those"
    }

    POSSESSIVE_PRONOUNS = {
        "his",
        "hers",
        "its",
        "their",
        "theirs",
        "nito",
        "niyan",
        "niyon",
    }

    FOLLOW_UP_PHRASES = {
        "continue",
        "tell me more",
        "explain more",
        "elaborate",
        "more",
        "go on",
        "next",
        "details",
        "more details",
        "give more details",
        "show more",
    }

    FOLLOW_UP_STARTERS = {
        "how",
        "where",
        "when",
        "why",
        "which",
        "what",
        "who",
        "ano",
        "anong",
        "sino",
        "kailan",
        "saan",
        "bakit",
        "paano",
        "alin",
    }

    GENERIC_FOLLOW_UP_TERMS = {
        # ==================================================
        # GENERAL / POLICY / MANUAL TERMS
        # ==================================================
        "eligible",
        "eligibility",
        "requirement",
        "requirements",
        "approval",
        "approver",
        "days",
        "how many days",
        "how many",
        "amount",
        "limit",
        "limits",
        "process",
        "procedure",
        "procedures",
        "policy",
        "policies",
        "rules",
        "rule",
        "benefit",
        "benefits",
        "entitled",
        "entitlement",
        "allowed",
        "allowance",
        "scope",
        "purpose",
        "definition",
        "overview",
        "summary",
        "meaning",
        "details",
        "role",
        "roles",

        # ==================================================
        # IT / CODING / TECHNICAL MANUAL TERMS
        # ==================================================
        "installation",
        "install",
        "setup",
        "configuration",
        "configure",
        "config",
        "settings",
        "environment",
        "dependency",
        "dependencies",
        "version",
        "versions",
        "syntax",
        "parameter",
        "parameters",
        "argument",
        "arguments",
        "option",
        "options",
        "command",
        "commands",
        "script",
        "scripts",
        "function",
        "functions",
        "method",
        "methods",
        "class",
        "classes",
        "module",
        "modules",
        "api",
        "endpoint",
        "endpoints",
        "request",
        "response",
        "payload",
        "database",
        "table",
        "field",
        "fields",
        "column",
        "columns",
        "schema",
        "query",
        "error",
        "errors",
        "error code",
        "error codes",
        "exception",
        "exceptions",
        "troubleshooting",
        "debug",
        "debugging",
        "log",
        "logs",
        "warning",
        "warnings",
        "security",
        "authentication",
        "authorization",
        "permission",
        "permissions",
        "access",
        "standard",
        "standards",
        "guideline",
        "guidelines",
        "coding standard",
        "coding standards",
        "naming",
        "naming convention",
        "convention",
        "conventions",
        "rule id",
        "rule number",

        # ==================================================
        # BIOGRAPHY / PERSON TEST TERMS
        # ==================================================
        "born",
        "birth",
        "birthplace",
        "place of birth",
        "date of birth",
        "died",
        "death",
        "president",
        "prime minister",
        "nationality",
        "occupation",
        "known for",
        "works",
        "legacy",
        "contribution",
        "contributions",
        "accomplishment",
        "accomplishments",
        "achievement",
        "achievements",
        "importance",
        "significance",
        "early life",
        "education",
        "career",
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

        question = question.strip()

        if not question:

            return question

        # ======================================
        # Fresh/current standalone request
        # ======================================
        # Do not attach an older conversation topic to freshness questions.
        # They must either be answered from current supported evidence or
        # safely fall back, never inherit a stale Rule/switch subject.
        if re.search(
            r"\b(?:latest|newest|current|today|this\s+year|released\s+this\s+year|most\s+recent)\b",
            question,
            flags=re.IGNORECASE,
        ):
            return question

        # ======================================
        # Explicit structured reference in the CURRENT question
        #
        # A direct Rule / Directive / Section identifier always wins over
        # conversation context. This prevents a question such as
        # ``What does Rule 99.99 say?`` from being rewritten against the
        # previous topic merely because it is short.
        # ======================================
        if extract_structured_reference(question) is not None:
            return question

        topic = self._get_current_or_history_topic(
            history_messages
        )

        # ======================================
        # Structured-reference follow-up
        #
        # Preserve the most recent exact Rule / Directive / Section
        # identifier when the user says "that rule", "this directive",
        # "that section", etc. This takes priority over the broader
        # document topic (for example, MISRA) so the referent is not lost.
        # ======================================
        structured_followup = self._resolve_structured_reference_followup(
            history_messages,
            question
        )

        if structured_followup != question:

            return structured_followup

        # ======================================
        # Structured attribute/pronoun follow-up
        #
        # Prefer the latest exact identifier for terse follow-ups before the
        # broad topic extractor sees generic phrases such as "what category".
        # The helper has its own conservative vocabulary guard, so a newly
        # named subject is left untouched for the normal new-topic path below.
        # ======================================
        structured_attribute_followup = (
            self._resolve_structured_attribute_followup(
                history_messages,
                question
            )
        )

        if structured_attribute_followup != question:
            return structured_attribute_followup

        # ======================================
        # Canonical object-relation follow-ups BEFORE new-topic detection
        #
        # Query normalization may intentionally compress a follow-up into a
        # relation phrase such as ``countries that signed it`` or
        # ``founder nito``. Those phrases no longer begin with an
        # interrogative, so resolve their explicit pronoun here before the
        # generic topic extractor can misclassify the whole phrase as a new
        # topic. Keep the grammar narrow so a genuinely new named subject
        # still wins later.
        # ======================================
        if re.search(
            r"\b(?:it|its|this|that|ito|iyan|iyon|nito|niyan|niyon)\b",
            question,
            flags=re.IGNORECASE,
        ) and re.match(
            r"^\s*(?:countries?\s+that\s+signed|signatories?\s+of|founders?|co[-\s]?founders?)\b",
            question,
            flags=re.IGNORECASE,
        ):
            object_referent = self._latest_explicit_object_referent(
                history_messages
            )
            if object_referent:
                return self._replace_pronouns(
                    question,
                    object_referent
                )

        # ======================================
        # Explicit new topic detection BEFORE pronoun replacement
        #
        # A question can name its own subject and still contain a pronoun,
        # for example: ``Explain who Jose Rizal was and why he was
        # significant.``  In that case the explicit current subject must
        # remain authoritative; ``he`` must not be replaced with the prior
        # conversation topic.
        # ======================================
        new_topic = self.topic_extractor.extract_new_topic(
            question
        )

        new_topic = self._clean_topic(
            new_topic
        )

        if (
            new_topic
            and not self._is_generic_followup_topic(
                new_topic
            )
        ):
            # A multi-word explicitly named subject must override an older
            # conversation topic even when the phrase contains a generic word
            # such as "policy", "procedure", "system", or "rule".  Earlier
            # builds could misclassify e.g. "the employee leave policy" as a
            # contextual follow-up simply because "policy" is in the generic
            # follow-up vocabulary, causing stale-topic rewrites such as
            # "the employee leave policy about MISRA".
            if self._looks_like_explicit_new_subject_phrase(
                question,
                new_topic,
            ):
                return question

            # Some short relation questions are falsely surfaced by the
            # generic topic extractor as if the whole interrogative were a
            # new topic, for example ``What position did he hold?``.  Do not
            # let that false-positive bypass typed person-reference recovery.
            # Explicit current subjects still win because this helper is
            # intentionally limited to clear pronoun-follow-up grammar.
            if not (
                self._is_clear_person_pronoun_followup(question)
                or self._is_clear_object_pronoun_followup(question)
                or (topic and self._is_contextual_followup(question))
            ):
                return question

        # ======================================
        # Typed person-reference follow-up
        #
        # ``current_topic`` can legitimately drift to a broader retrieved
        # document/topic after a relation question. Person pronouns must not
        # inherit that broad topic when the conversation already contains an
        # explicit person referent such as ``Who is Emilio Aguinaldo?``.
        # Resolve only gendered/person pronouns here; object pronouns such as
        # ``it`` continue through the existing structured/topic logic below.
        # ======================================
        if self._contains_person_pronoun(question):
            person_referent = self._latest_explicit_person_referent(
                history_messages
            )

            if person_referent:
                return self._replace_person_pronouns(
                    question,
                    person_referent
                )

        # ======================================
        # Pronoun-based follow-up
        #
        # Examples:
        # what did he do?
        # where was he born?
        # what are his works?
        # what is it for?
        # what are its rules?
        # how does it work?
        # ======================================
        if self._contains_pronoun(
            question
        ):

            # Object/possessive pronouns are inherently reference-dependent.
            # Try the latest explicitly named object even after normalization
            # has removed an interrogative prefix (for example Tagalog
            # ``Sino ang ... nito?`` -> ``founder nito``). This also keeps
            # treaty follow-ups such as ``Which countries were parties to it?``
            # bound to the exact treaty named in the preceding user turn.
            if re.search(
                r"\b(?:it|its|this|that|these|those|ito|iyan|iyon|nito|niyan|niyon)\b",
                str(question),
                flags=re.IGNORECASE,
            ):
                object_referent = self._latest_explicit_object_referent(
                    history_messages
                )
                if object_referent:
                    return self._replace_pronouns(
                        question,
                        object_referent
                    )

            if topic:

                return self._replace_pronouns(
                    question,
                    topic
                )

            return question

        # ======================================
        # Short follow-up command
        #
        # Examples:
        # more
        # tell me more
        # details
        # explain more
        # ======================================
        if topic:

            followup = self._handle_followup_command(
                question,
                topic
            )

            if followup != question:

                return followup

        # ======================================
        # Contextual follow-up without pronoun
        #
        # Examples:
        # when was born?
        # known for?
        # installation steps
        # syntax
        # rule 10.1
        # requirements
        # ======================================
        if topic and self._is_contextual_followup(
            question
        ):

            return self._attach_topic(
                question,
                topic
            )

        return question


    def _looks_like_explicit_new_subject_phrase(
        self,
        question,
        topic,
    ):
        """Return True for a self-contained newly named multi-word subject.

        This is intentionally grammar-based rather than domain-specific.  A
        phrase such as ``employee leave policy`` contains one generic noun but
        also names a concrete subject through its descriptive modifiers.  By
        contrast ``approval process`` or ``more requirements`` are composed
        only of generic follow-up vocabulary and remain context-dependent.
        """

        clean_question = re.sub(
            r"\s+",
            " ",
            str(question or "").strip().lower(),
        ).strip(" ?.!,:;")
        clean_topic = re.sub(
            r"\s+",
            " ",
            str(topic or "").strip().lower(),
        ).strip(" ?.!,:;")

        if not clean_question or not clean_topic:
            return False

        # Deictic/reference words are a strong signal that the phrase still
        # depends on prior context rather than naming a fresh subject.
        if re.search(
            r"\b(?:he|him|his|she|her|hers|it|its|this|that|these|those|"
            r"ito|iyan|iyon|yan|yun|dito|diyan|doon|dun|nito|niyan|niyon|"
            r"siya|niya|same|prior|previous)\b",
            clean_question,
            flags=re.IGNORECASE,
        ):
            return False

        words = [
            word
            for word in re.findall(r"[a-z0-9]+", clean_topic)
            if word not in {"the", "a", "an", "our", "company"}
        ]
        if len(words) < 2:
            return False

        generic_words = set()
        for term in self.GENERIC_FOLLOW_UP_TERMS:
            generic_words.update(re.findall(r"[a-z0-9]+", term.lower()))
        generic_words.update({
            "more", "additional", "applicable", "same", "current",
            "previous", "prior", "document", "source", "code", "snippet",
        })

        meaningful = [word for word in words if word not in generic_words]
        return bool(meaningful)


    def _latest_structured_reference(
        self,
        history_messages,
        requested_kind=None
    ):

        """Return the most recent exact structured reference from user turns."""

        if not history_messages:
            return None

        for message in reversed(history_messages):
            if not isinstance(message, dict):
                continue

            if message.get("role") != "user":
                continue

            reference = extract_structured_reference(
                str(message.get("content", ""))
            )

            if reference is None:
                continue

            if requested_kind and reference.kind != requested_kind:
                continue

            return reference

        return None


    def _resolve_structured_reference_followup(
        self,
        history_messages,
        question
    ):

        """Resolve phrases such as ``that rule`` to the latest exact ID.

        The lookup is generic for Rule, Directive/Dir, and Section references
        and scans only prior user messages. It does not infer an identifier
        when the conversation never contained one.
        """

        if not question or not history_messages:
            return question

        clean = question.strip()

        reference_terms = {
            "rule": ("rule",),
            "directive": ("directive", "dir"),
            "section": ("section",),
            "article": ("article",),
            "chapter": ("chapter",),
            "part": ("part",),
        }

        requested_kind = None

        for kind, terms in reference_terms.items():
            term_pattern = "|".join(
                re.escape(term)
                for term in terms
            )

            if re.search(
                rf"\b(?:that|this|the|same|previous)\s+(?:{term_pattern})\b",
                clean,
                flags=re.IGNORECASE
            ):
                requested_kind = kind
                break

        if requested_kind is None:
            return question

        latest_reference = self._latest_structured_reference(
            history_messages,
            requested_kind=requested_kind
        )

        if latest_reference is None:
            return question

        if latest_reference.kind == "directive":
            replacement = f"Directive {latest_reference.identifier}"
        else:
            replacement = latest_reference.display_name

        term_pattern = "|".join(
            re.escape(term)
            for term in reference_terms[requested_kind]
        )

        resolved = re.sub(
            rf"\b(?:that|this|the|same|previous)\s+(?:{term_pattern})\b",
            replacement,
            clean,
            count=1,
            flags=re.IGNORECASE
        )

        return self._clean_resolved_question(
            resolved
        )

    def _resolve_structured_attribute_followup(
        self,
        history_messages,
        question
    ):

        """Prefer the latest exact Rule/Directive/Section for terse follow-ups.

        The normal conversation topic can be a broad document name such as
        ``MISRA`` even after the user just asked about ``Rule 13.5``. For
        compact pronoun/attribute follow-ups, the latest explicit structured
        identifier is the more precise referent.

        This runs only after explicit-new-topic detection, so a newly named
        subject still wins and cannot be overwritten by older conversation
        context.
        """

        if not question or not history_messages:
            return question

        reference = self._latest_structured_reference(
            history_messages
        )

        if reference is None:
            return question

        clean = re.sub(
            r"\s+",
            " ",
            str(question).strip()
        )
        lowered = clean.casefold().rstrip("?.!")

        if not lowered:
            return question

        words = re.findall(r"\b\w+\b", lowered)

        # Resolver input is normalized, so use a conservative vocabulary
        # instead of capitalization/NER to distinguish terse follow-ups from
        # a newly named subject. If an unfamiliar content token is present,
        # leave the question untouched so normal new-topic detection can win.
        structured_followup_vocabulary = {
            "what", "which", "why", "how", "category", "does", "do", "is",
            "are", "was", "were", "it", "its", "this", "that", "the",
            "same", "previous", "belong", "belongs", "to", "apply", "applies",
            "applicable", "applicability", "rationale", "reason", "important",
            "importance", "example", "examples", "analysis", "scope",
            "requirement", "requirements", "meaning", "detail", "details",
            "explain", "describe", "more", "for", "about", "give", "show",
            "me", "mentioned", "in", "as", "well", "rule", "directive",
            "dir", "section", "and", "or", "please", "assigned", "assign",
            "cover", "covers", "covered", "version", "versions", "standard",
            "standards", "given", "exist", "exists", "stated", "state",
            "say", "says", "discuss", "discusses", "discussed",
            "amplification", "exception", "exceptions", "see", "also",
            "article", "chapter", "part",
            "portability", "portable", "decidability", "decidable", "undecidable",
            "distinguish", "distinguishes", "difference", "categories", "mandatory",
            "required", "advisory", "c",
            "ano", "anong", "bakit", "paano", "kategorya", "dahilan",
            "halimbawa", "saklaw", "ipaliwanag", "ito", "iyan", "nito",
            "niyan", "ang", "ng", "sa",
        }

        if words and any(
            word not in structured_followup_vocabulary
            for word in words
        ):
            return question

        structured_detail_cue = bool(
            re.search(
                r"\b(?:category|rationale|reason|important|importance|"
                r"example|examples|analysis|applies|apply|applicable|"
                r"applicability|scope|requirement|requirements|meaning|"
                r"detail|details|explain|describe|assigned|cover|covers|covered|"
                r"version|versions|standard|standards|decidability|decidable|undecidable|"
                r"amplification|exception|exceptions|see\s+also|say|says|discuss|discusses|"
                r"kategorya|dahilan|halimbawa|saklaw|ipaliwanag)\b",
                lowered,
                flags=re.IGNORECASE,
            )
        )

        pronoun_present = bool(
            re.search(
                r"\b(?:it|its|this|that)\b",
                lowered,
                flags=re.IGNORECASE,
            )
        )

        # A very short pronoun-only follow-up is safe to bind to the latest
        # exact structured reference. Longer pronoun questions require a
        # structured-detail cue to avoid hijacking unrelated conversation.
        if pronoun_present and (
            structured_detail_cue
            or len(words) <= 4
        ):
            replacement = (
                f"Directive {reference.identifier}"
                if reference.kind == "directive"
                else reference.display_name
            )

            resolved = re.sub(
                r"\bits\b",
                lambda match: f"{replacement}'s",
                clean,
                count=1,
                flags=re.IGNORECASE,
            )

            if resolved == clean:
                resolved = re.sub(
                    r"\b(?:it|this|that)\b",
                    replacement,
                    clean,
                    count=1,
                    flags=re.IGNORECASE,
                )

            return self._clean_resolved_question(
                resolved
            )

        # Bare attribute follow-ups such as "category?" or "the rationale?"
        # are attached only when they are short and clearly structured.
        if structured_detail_cue and len(words) <= 5:
            replacement = (
                f"Directive {reference.identifier}"
                if reference.kind == "directive"
                else reference.display_name
            )
            return self._clean_resolved_question(
                f"{clean} for {replacement}"
            )

        return question


    def _is_clear_person_pronoun_followup(
        self,
        question
    ):

        """Return True for relation questions whose grammatical subject is a person pronoun.

        This guard exists only to distinguish a genuine follow-up such as
        ``What position did he hold?`` from a current question that explicitly
        names its own subject.  It is domain-neutral and does not infer person
        identity from retrieved documents or topic titles.
        """

        if not self._contains_person_pronoun(question):
            return False

        clean = re.sub(
            r"\s+",
            " ",
            str(question).strip()
        )

        # Interrogative + relation phrase + auxiliary + person pronoun.
        # Examples: ``What position did he hold?`` /
        # ``Which office did she occupy?`` / ``What title does he have?``.
        if re.search(
            r"^\s*(?:what|which|who|where|when|why|how|ano|anong|alin|saan|kailan|bakit|paano)\b"
            r".*\b(?:do|does|did|is|are|was|were|has|have|had|can|could|will|would|should)\s+"
            r"(?:he|she|him|her|siya)\b",
            clean,
            flags=re.IGNORECASE,
        ):
            return True

        # Possessive short follow-ups such as ``What is his position?`` are
        # also clearly dependent on the prior person referent.
        if re.search(
            r"^\s*(?:what|which|who|where|when|why|how|ano|anong|alin|saan|kailan|bakit|paano)\b"
            r".*\b(?:his|hers|her)\b",
            clean,
            flags=re.IGNORECASE,
        ):
            return True

        # Broader interrogative follow-ups such as ``What additional role is
        # listed for him?`` still have a grammatical person-pronoun referent.
        if re.search(
            r"^\s*(?:what|which|who|where|when|why|how|ano|anong|alin|saan|kailan|bakit|paano)\b"
            r".*\b(?:he|him|his|she|her|hers|siya)\b",
            clean,
            flags=re.IGNORECASE,
        ):
            return True

        return False


    def _is_clear_object_pronoun_followup(self, question):

        if not question:
            return False

        clean = re.sub(r"\s+", " ", str(question).strip())

        return bool(
            re.search(
                r"^\s*(?:what|which|who|where|when|why|how|ano|anong|alin|sino|saan|kailan|bakit|paano)\b"
                r".*\b(?:it|its|this|that|these|those|ito|iyan|iyon|nito|niyan|niyon)\b",
                clean,
                flags=re.IGNORECASE,
            )
        )


    def _latest_explicit_person_referent(
        self,
        history_messages
    ):

        """Return the latest person explicitly introduced by the user.

        This is intentionally domain-neutral and conservative. A person is
        remembered only when the user's own wording explicitly uses a person
        identity form such as ``Who is X?``, ``Who was X?`` or ``Sino si X?``.
        Retrieved document titles are never promoted to a person referent.
        """

        if not history_messages:
            return None

        patterns = (
            r"^\s*who\s+(?:is|was)\s+(.+?)\s*[?!.]*$",
            r"^\s*sino\s+si\s+(.+?)\s*[?!.]*$",
            r"^\s*(?:give\s+(?:me\s+)?(?:a\s+)?(?:short|brief|concise)?\s*profile\s+of|briefly\s+describe|describe)\s+(.+?)(?:\s+briefly)?\s*[?!.]*$",
        )

        for message in reversed(history_messages):
            if not isinstance(message, dict):
                continue

            if message.get("role") != "user":
                continue

            content = re.sub(
                r"\s+",
                " ",
                str(message.get("content", "")).strip()
            )

            if not content:
                continue

            for pattern in patterns:
                match = re.match(
                    pattern,
                    content,
                    flags=re.IGNORECASE
                )

                if not match:
                    continue

                candidate_text = re.split(
                    r"\s+(?:and|at)\s+(?=(?:who|what|when|where|why|how|which|"
                    r"sino|ano|kailan|saan|bakit|paano|alin|explain|ipaliwanag)\b)",
                    match.group(1),
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0]

                candidate = self._clean_topic(
                    candidate_text
                )

                if (
                    candidate
                    and not self._is_generic_followup_topic(candidate)
                ):
                    return candidate

        return None

    def _latest_explicit_object_referent(
        self,
        history_messages
    ):
        """Recover a compact explicitly named object for pronoun follow-ups.

        This is intentionally conservative and is used only when the current
        question contains an object/possessive pronoun. It helps Tagalog forms
        such as ``Ano ang Katipunan sa maikling paliwanag?`` followed by
        ``Sino ang isa sa mga nagtatag nito?`` without promoting the whole
        explanatory phrase to the topic.
        """

        if not history_messages:
            return None

        patterns = (
            r"^\s*ano\s+ang\s+(.+?)(?:\s+sa\s+(?:maikling\s+paliwanag|simpleng\s+paliwanag|kabuuan))?\s*[?!.]*$",
            r"^\s*what\s+is\s+(.+?)(?:\s+in\s+(?:brief|short|simple\s+terms))?\s*[?!.]*$",
            r"^\s*give\s+(?:me\s+)?(?:the\s+)?signing\s+date\s+of\s+(.+?)\s*[?!.]*$",
            r"^\s*where\s+was\s+(.+?)\s+signed\s*[?!.]*$",
            r"^\s*when\s+did\s+(.+?)\s+become\s+effective\s*[?!.]*$",
        )

        for message in reversed(history_messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = re.sub(r"\s+", " ", str(message.get("content", "")).strip())
            if not content or extract_structured_reference(content) is not None:
                continue
            for pattern in patterns:
                match = re.match(pattern, content, flags=re.IGNORECASE)
                if not match:
                    continue
                candidate = self._clean_topic(match.group(1))
                if not candidate:
                    continue
                words = candidate.split()
                if len(words) > 6 or self._is_generic_followup_topic(candidate):
                    continue
                # Prefer an explicit named noun phrase, not a generic question
                # remainder such as "main purpose".
                if not any(ch.isupper() for ch in match.group(1)):
                    continue
                return candidate
        return None


    def _contains_person_pronoun(
        self,
        question
    ):

        if not question:
            return False

        return bool(
            re.search(
                r"\b(?:he|him|his|she|her|hers|siya)\b",
                str(question),
                flags=re.IGNORECASE
            )
        )

    def _replace_person_pronouns(
        self,
        question,
        person_referent
    ):

        """Replace only person pronouns with the latest explicit person."""

        resolved = str(question).strip()

        resolved = re.sub(
            r"\bher\s+(\w+)",
            lambda match: f"{person_referent}'s {match.group(1)}",
            resolved,
            count=1,
            flags=re.IGNORECASE
        )

        resolved = re.sub(
            r"\b(?:his|hers)\b",
            lambda match: f"{person_referent}'s",
            resolved,
            count=1,
            flags=re.IGNORECASE
        )

        resolved = re.sub(
            r"\b(?:he|him|she|her|siya)\b",
            person_referent,
            resolved,
            count=1,
            flags=re.IGNORECASE
        )

        return self._clean_resolved_question(
            resolved
        )


    def _get_current_or_history_topic(
        self,
        history_messages
    ):

        topic = ChatManager.get_current_topic()

        topic = self._clean_topic(
            topic
        )

        if topic:

            return topic

        topic = self.topic_extractor.extract(
            history_messages
        )

        topic = self._clean_topic(
            topic
        )

        if topic:

            # Use only for resolving the current question.
            # AnswerService handles permanent topic updates.
            return topic

        return None

    def _contains_pronoun(
        self,
        question
    ):

        clean = (
            question
            .lower()
            .strip()
        )

        words = re.findall(
            r"\b\w+\b",
            clean
        )

        pronouns = (
            self.PERSON_PRONOUNS
            | self.OBJECT_PRONOUNS
            | self.PLURAL_PRONOUNS
            | self.POSSESSIVE_PRONOUNS
        )

        for word in words:

            if word in pronouns:

                return True

        return False

    def _replace_pronouns(
        self,
        question,
        topic
    ):

        resolved = question.strip()

        # Protect relative-clause "that".
        #
        # Example:
        # ladies that had relationship with Jose Rizal
        #
        # In this case, "that" is not a follow-up pronoun.
        # It should NOT become:
        # ladies Jose Rizal had relationship...
        resolved = re.sub(
            r"\b(\w+)\s+that\s+(had|has|have|is|are|was|were|can|will|would|should|must|requires?|contains?|includes?|uses?|allows?|supports?|signs?|signed)\b",
            r"\1 __RELATIVE_THAT__ \2",
            resolved,
            flags=re.IGNORECASE
        )

        # ======================================
        # Possessive "her"
        #
        # Examples:
        # what is her role?
        # -> what is Maria Clara's role?
        #
        # tell me about her early life
        # -> tell me about Maria Clara's early life
        # ======================================
        resolved = re.sub(
            r"\bher\s+(\w+)",
            lambda match:
                f"{topic}'s {match.group(1)}",
            resolved,
            count=1,
            flags=re.IGNORECASE
        )

        # ======================================
        # Tagalog object/possessive references
        #
        # ``nito`` / ``niyan`` are relational ("of it/that"), so keep the
        # grammar natural enough for the multilingual canonicalizer instead
        # of turning them into an English apostrophe inside a Tagalog query.
        # ======================================
        resolved = re.sub(
            r"\b(?:nito|niyan|niyon)\b",
            lambda match: f"ng {topic}",
            resolved,
            count=1,
            flags=re.IGNORECASE
        )

        resolved = re.sub(
            r"\b(?:ito|iyan|iyon)\b",
            topic,
            resolved,
            count=1,
            flags=re.IGNORECASE
        )

        # ======================================
        # Possessive pronouns
        #
        # Examples:
        # what are his works?
        # -> what are Jose Rizal's works?
        #
        # what are its rules?
        # -> what are MISRA's rules?
        # ======================================
        resolved = re.sub(
            r"\b(his|hers|its|their|theirs)\b",
            lambda match:
                f"{topic}'s",
            resolved,
            count=1,
            flags=re.IGNORECASE
        )

        # ======================================
        # Subject/object pronouns
        #
        # Examples:
        # what did he do?
        # -> what did Emilio Aguinaldo do?
        #
        # what is it for?
        # -> what is MISRA for?
        # ======================================
        pronoun_pattern = (
            r"\b("
            r"he|him|she|her|siya|it|this|that|they|them|these|those"
            r")\b"
        )

        resolved = re.sub(
            pronoun_pattern,
            lambda match:
                topic,
            resolved,
            count=1,
            flags=re.IGNORECASE
        )

        resolved = resolved.replace(
            "__RELATIVE_THAT__",
            "that"
        )

        resolved = self._clean_resolved_question(
            resolved
        )

        return resolved

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

    def _is_contextual_followup(
        self,
        question
    ):

        clean = (
            question
            .lower()
            .strip()
            .rstrip("?.!")
        )

        if not clean:

            return False

        words = clean.split()

        if not words:

            return False

        first_word = words[0]

        # Question-style contextual follow-ups must remain short.
        #
        # Examples that should use the current topic:
        # - when effective
        # - where configured
        # - what are the requirements
        #
        # Long questions usually contain their own explicit subject
        # and must not inherit an unrelated previous topic.
        #
        # Pronoun-based questions are already resolved before this
        # method is called.
        if first_word in self.FOLLOW_UP_STARTERS:

            return len(words) <= 5

        # Exact generic follow-up:
        # syntax
        # requirements
        # known for
        # early life
        if clean in self.GENERIC_FOLLOW_UP_TERMS:

            return True

        # Rule/error follow-up:
        # rule 10.1
        # error code 500
        if self._looks_like_rule_or_error_followup(
            clean
        ):

            return True

        # Short attribute-style follow-up:
        # installation steps
        # approval process
        # known for
        # early life
        if len(words) <= 5:

            if self._contains_generic_followup_term(
                clean
            ):

                return True

        return False

    def _contains_generic_followup_term(
        self,
        clean
    ):

        if clean in self.GENERIC_FOLLOW_UP_TERMS:

            return True

        words = clean.split()

        for word in words:

            if word in self.GENERIC_FOLLOW_UP_TERMS:

                return True

        for term in self.GENERIC_FOLLOW_UP_TERMS:

            if " " in term and term in clean:

                return True

        return False

    def _looks_like_rule_or_error_followup(
        self,
        clean
    ):

        patterns = [
            r"^rule\s+\d+(\.\d+)*$",
            r"^misra\s+\d+(\.\d+)*$",
            r"^misra\s+rule\s+\d+(\.\d+)*$",
            r"^error\s+\d+$",
            r"^error\s+code\s+\d+$",
            r"^code\s+\d+$",
        ]

        for pattern in patterns:

            if re.match(
                pattern,
                clean
            ):

                return True

        return False

    def _attach_topic(
        self,
        question,
        topic
    ):

        clean_question = (
            question
            .strip()
            .rstrip(".?!")
        )

        clean_topic = self._clean_topic(
            topic
        )

        if not clean_topic:

            return question

        # Avoid duplicate topic attachment.
        if clean_topic.lower() in clean_question.lower():

            return question

        return f"{clean_question} about {clean_topic}"

    def _clean_topic(
        self,
        topic
    ):

        if not topic:

            return None

        topic = str(topic).strip()

        if not topic:

            return None

        topic = re.sub(
            r"[?!.]+$",
            "",
            topic
        ).strip()

        # Remove common question prefixes accidentally captured as topic.
        topic = re.sub(
            r"^(who|what|where|when|why|how)\s+",
            "",
            topic,
            flags=re.IGNORECASE
        ).strip()

        topic = re.sub(
            r"^(is|are|was|were|about)\s+",
            "",
            topic,
            flags=re.IGNORECASE
        ).strip()

        topic = re.sub(
            r"\s+",
            " ",
            topic
        ).strip()

        if not topic:

            return None

        return topic

    def _is_generic_followup_topic(
        self,
        topic
    ):

        clean = (
            topic
            .lower()
            .strip()
            .rstrip("?.!")
        )

        if not clean:

            return True

        if clean in self.GENERIC_FOLLOW_UP_TERMS:

            return True

        if clean in self.FOLLOW_UP_PHRASES:

            return True

        # Single generic word should not become a new topic.
        if (
            len(clean.split()) == 1
            and self._contains_generic_followup_term(
                clean
            )
        ):

            return True

        return False

    def _clean_resolved_question(
        self,
        question
    ):

        question = re.sub(
            r"\s+",
            " ",
            question
        ).strip()

        question = re.sub(
            r"\s+([?.!,;:])",
            r"\1",
            question
        )

        return question