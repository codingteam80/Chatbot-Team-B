import re

from chat.topic_extractor import TopicExtractor
from chat.chat_manager import ChatManager


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
        "her"
    }

    OBJECT_PRONOUNS = {
        "it",
        "this",
        "that"
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
        "theirs"
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

        topic = self._get_current_or_history_topic(
            history_messages
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
        # Explicit new topic detection
        #
        # Important:
        # This must happen before contextual follow-up.
        #
        # Example:
        # Current topic: Jose Rizal
        # User: what is MISRA?
        #
        # This should become a new topic, not:
        # what is MISRA about Jose Rizal
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

            # Do not save the topic yet.
            # AnswerService will save the canonical topic
            # only after a valid answer is generated.
            return question

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
            r"\b(\w+)\s+that\s+(had|has|have|is|are|was|were|can|will|would|should|must|requires?|contains?|includes?|uses?|allows?|supports?)\b",
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
            r"he|him|she|her|it|this|that|they|them|these|those"
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

        # Question-style follow-up:
        # what did he do is already handled by pronoun resolver.
        # what are the requirements
        # when effective
        # where configured
        if first_word in self.FOLLOW_UP_STARTERS:

            return True

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