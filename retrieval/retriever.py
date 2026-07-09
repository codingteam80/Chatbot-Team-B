print("USING retriever.py")

import re
import unicodedata

from retrieval.bm25_index import BM25Searcher
from retrieval.chroma_search import ChromaSearcher
from retrieval.hybrid_search import HybridRetriever

from config.settings import (
    BM25_TOP_K,
    VECTOR_TOP_K,
    FINAL_TOP_K,
    DEBUG_RETRIEVAL,
    ENABLE_RERANKER,
    COMPLETENESS_TOP_K,
    CONTEXT_EXPANSION_PREVIOUS_CHUNKS,
    CONTEXT_EXPANSION_NEXT_CHUNKS,
    CONTEXT_EXPANSION_MAX_SEEDS,
    MIN_RETRIEVAL_SCORE
)


class CompanyRetriever:

    def __init__(self):

        self.bm25 = BM25Searcher()
        self.chroma = ChromaSearcher()
        self.hybrid = HybridRetriever()

        self.reranker = None

        if ENABLE_RERANKER:

            from retrieval.reranker import CrossEncoderReranker
            self.reranker = CrossEncoderReranker()

    def _normalize_text(self, text):

        if not text:
            return ""

        text = text.lower()

        text = unicodedata.normalize(
            "NFKD",
            text
        )

        text = "".join(
            char for char in text
            if not unicodedata.combining(char)
        )

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        return text

    def _is_short_lookup_query(self, query):

        if not query:
            return False

        words = query.strip().split()

        return len(words) <= 5

    def _is_list_or_relationship_query(
        self,
        query
    ):

        """
        Detect genuine list, multi-answer, procedure,
        and relationship questions.

        Important:
        Only inspect the beginning of the enriched query
        for list/completeness terms.

        This prevents later enrichment terms such as:
            requirements
            rules
            steps
            configuration

        from incorrectly activating list mode.
        """

        if not query:

            return False

        clean = self._normalize_text(
            query
        )

        if not clean:

            return False

        tokens = clean.split()

        # The original normalized question is always
        # placed at the beginning of the enriched query.
        #
        # Inspect only the first few words when deciding
        # whether this is a list/completeness request.
        prefix_tokens = tokens[:8]

        intent_prefix = " ".join(
            prefix_tokens
        )

        # Explicit list-style question starters.
        explicit_patterns = [
            r"^(who are|what are)\b",
            r"^(list|enumerate)\b",
            r"^name the\b",
            r"^give me the list\b",
            r"^(sino sino|ano ano|ilista)\b",
            r"^(all|examples|types|categories)\b",

            # Explicit procedure requests
            r"^how to\b",
            r"^step by step\b",
            r"^steps\b",
            r"^procedure for\b",
            r"^procedures for\b",
            r"^process for\b",
            r"^workflow for\b",
        ]

        for pattern in explicit_patterns:

            if re.search(
                pattern,
                intent_prefix
            ):

                return True

        # Completeness terms must appear near the start
        # of the original question, not only in enrichment.
        completeness_terms = {
            "steps",
            "requirements",
            "rules",
            "conditions",
            "exceptions",
            "approvers",
            "parameters",
            "options",
            "commands",
            "errors",
        }

        if set(prefix_tokens).intersection(
            completeness_terms
        ):

            return True

        # Relationship questions may contain their
        # identifying terms later in the sentence.
        relationship_patterns = [
            r"\bladies\b",
            r"\bwomen\b",
            r"\brelationship\b",
            r"\brelationships\b",
            r"\bromantic relationship\b",
            r"\bromantic relationships\b",
            r"\blove interest\b",
            r"\blove interests\b",
            r"\bgirlfriend\b",
            r"\bgirlfriends\b",
            r"\bpersonal life\b",
            r"\bassociated women\b",
            r"\bconnected to\b",
            r"\bassociated with\b",
        ]

        for pattern in relationship_patterns:

            if re.search(
                pattern,
                clean
            ):

                return True

        return False

    def _final_top_k_for_query(
        self,
        query
    ):

        """
        Use more final chunks for completeness-style questions.
        Keep normal questions concise.
        """

        if self._is_list_or_relationship_query(
            query
        ):

            return max(
                FINAL_TOP_K,
                COMPLETENESS_TOP_K
            )

        return FINAL_TOP_K

    def _chunk_key(
        self,
        metadata
    ):

        """
        Build a stable key for a chunk using file path/name + chunk_id.
        """

        if not metadata:

            return None

        file_key = (
            metadata.get("file_path")
            or metadata.get("file_name")
        )

        chunk_id = metadata.get(
            "chunk_id"
        )

        if file_key is None or chunk_id is None:

            return None

        try:

            chunk_id = int(
                chunk_id
            )

        except:

            return None

        return f"{file_key}::{chunk_id}"

    def _get_neighbor_lookup(
        self
    ):

        """
        Build lookup of all indexed chunks from BM25 records.

        BM25 stores the original records, so we can use it
        to fetch chunk_id - 1 / chunk_id + 1 / chunk_id + 2.
        """

        if hasattr(
            self,
            "_neighbor_lookup"
        ):

            return self._neighbor_lookup

        lookup = {}

        records = getattr(
            self.bm25,
            "records",
            []
        )

        for record in records:

            metadata = record.get(
                "metadata",
                {}
            )

            key = self._chunk_key(
                metadata
            )

            if key:

                lookup[key] = record

        self._neighbor_lookup = lookup

        return self._neighbor_lookup

    def _record_to_result(
        self,
        record,
        base_score=0.0,
        expanded_from=None
    ):

        """
        Convert a stored BM25 record back to retrieval result format.
        """

        metadata = dict(
            record.get(
                "metadata",
                {}
            )
        )

        return {
            "text": record.get(
                "text",
                ""
            ),
            "metadata": metadata,
            "score": float(
                base_score
            ),
            "_expanded_neighbor": True,
            "_expanded_from": expanded_from,
        }

    def _expand_with_neighbor_chunks(
        self,
        results
    ):

        """
        Generic context completeness expansion.

        For list/procedure/requirement/rule questions,
        the relevant answer may continue in nearby chunks.

        Example:
        - chunk 10 has the section heading
        - chunk 11 has items 1-5
        - chunk 12 has items 6-10

        This method adds nearby chunks from the same file.
        """

        if not results:

            return results

        lookup = self._get_neighbor_lookup()

        if not lookup:

            return results

        expanded = []
        seen = set()

        seed_results = results[
            :CONTEXT_EXPANSION_MAX_SEEDS
        ]

        for seed in seed_results:

            metadata = seed.get(
                "metadata",
                {}
            )

            file_key = (
                metadata.get("file_path")
                or metadata.get("file_name")
            )

            try:

                chunk_id = int(
                    metadata.get(
                        "chunk_id"
                    )
                )

                total_chunks = int(
                    metadata.get(
                        "total_chunks",
                        chunk_id + 1
                    )
                )

            except:

                key = self._chunk_key(
                    metadata
                )

                if key and key not in seen:

                    seen.add(
                        key
                    )

                    expanded.append(
                        seed
                    )

                continue

            start_chunk = max(
                0,
                chunk_id - CONTEXT_EXPANSION_PREVIOUS_CHUNKS
            )

            end_chunk = min(
                total_chunks - 1,
                chunk_id + CONTEXT_EXPANSION_NEXT_CHUNKS
            )

            for neighbor_id in range(
                start_chunk,
                end_chunk + 1
            ):

                neighbor_metadata = dict(
                    metadata
                )

                neighbor_metadata["chunk_id"] = neighbor_id

                neighbor_key = self._chunk_key(
                    neighbor_metadata
                )

                if not neighbor_key:

                    continue

                if neighbor_key in seen:

                    continue

                record = lookup.get(
                    neighbor_key
                )

                if record:

                    seen.add(
                        neighbor_key
                    )

                    if neighbor_id == chunk_id:

                        seed["_expanded_neighbor"] = False

                        expanded.append(
                            seed
                        )

                    else:

                        expanded.append(
                            self._record_to_result(
                                record=record,
                                base_score=seed.get(
                                    "score",
                                    0.0
                                ) * 0.95,
                                expanded_from=chunk_id
                            )
                        )

        if DEBUG_RETRIEVAL:

            print("\n===== NEIGHBOR CHUNK EXPANSION =====")
            print(f"Seeds used       : {len(seed_results)}")
            print(f"Before expansion : {len(results)}")
            print(f"After expansion  : {len(expanded)}")

            for item in expanded[:15]:

                metadata = item.get(
                    "metadata",
                    {}
                )

                print(
                    f"{metadata.get('file_name', 'Unknown')} "
                    f"chunk={metadata.get('chunk_id', '?')} "
                    f"expanded={item.get('_expanded_neighbor', False)}"
                )

            print("====================================\n")

        return expanded

    def _list_relationship_content_score(
        self,
        query,
        item
    ):

        """
        Score chunks for list / relationship questions.

        For relationship/list questions, intro biography chunks
        should not automatically win. Chunks with relationship,
        wife, romance, marriage, sweetheart, or similar terms
        should rank higher.
        """

        text = item.get(
            "text",
            ""
        )

        if not text:

            item["_list_score"] = 0.0

            return 0.0

        normalized = self._normalize_text(
            text
        )

        padded = f" {normalized} "

        score = float(
            item.get(
                "score",
                0.0
            )
        )

        score += self._query_match_score(
            query,
            text
        )

        positive_terms = {
            " relationship ": 1.20,
            " relationships ": 1.20,
            " romantic ": 1.20,
            " romance ": 1.20,
            " love ": 1.00,
            " lover ": 1.00,
            " sweetheart ": 1.40,
            " courtship ": 1.30,
            " courted ": 1.20,
            " girlfriend ": 1.20,
            " girlfriends ": 1.20,
            " wife ": 1.50,
            " common law wife ": 1.80,
            " married ": 1.20,
            " marriage ": 1.20,
            " fiancee ": 1.20,
            " fiance ": 1.20,
            " engaged ": 1.10,
            " women ": 0.90,
            " woman ": 0.90,
            " ladies ": 0.90,
            " lady ": 0.90,
            " personal life ": 0.90,

            # Useful in biography documents where relationship
            # information may appear as names, family terms,
            # or personal-life references rather than explicit
            # "relationship" wording.
            " common law ": 1.20,
            " partner ": 1.00,
            " spouse ": 1.00,
            " beloved ": 1.00,
            " affection ": 0.90,
            " suitor ": 0.90,
            " admirer ": 0.90,
            " daughter ": 0.70,
            " family ": 0.50,
        }

        positive_hit = False

        for term, weight in positive_terms.items():

            if term in padded:

                score += weight

                positive_hit = True

        negative_terms = [
            " monument ",
            " monuments ",
            " statue ",
            " park ",
            " popular culture ",
            " film ",
            " tv series ",
            " portrayed ",
            " actor ",
            " actress ",
            " frigate ",
            " ship ",
            " coin ",
            " bust ",
            " painting ",
            " portrait ",
            " opera ",
            " movie ",
            " awards ",
            " external links ",
            " references ",
            " bibliography ",
            " further reading ",
            " retrieved ",
            " archived ",
            " http ",
            " https ",
            " www ",
        ]

        for term in negative_terms:

            if term in padded:

                score -= 0.80

        section_signals = [
            " personal life ",
            " early life ",
            " education ",
            " women ",
            " relationship ",
            " relationships ",
            " marriage ",
            " wife ",
            " family ",
        ]

        if any(
            signal in padded
            for signal in section_signals
        ):

            score += 0.80

            positive_hit = True

        # If this is a relationship/list question and the chunk
        # has no relationship signal, lower its priority.
        if not positive_hit:

            score -= 1.00

        item["_list_score"] = score

        return score

    def _prioritize_list_relationship_chunks(
        self,
        query,
        results
    ):

        """
        Prioritize chunks that are useful for relationship/list answers.
        """

        if not results:

            return results

        ranked = sorted(
            results,
            key=lambda item:
                self._list_relationship_content_score(
                    query,
                    item
                ),
            reverse=True
        )

        useful_results = [
            item for item in ranked
            if item.get(
                "_list_score",
                0.0
            ) > -0.50
        ]

        if DEBUG_RETRIEVAL:

            print("\n===== LIST RELATIONSHIP RANKING =====")

            for item in ranked[:15]:

                preview = (
                    item.get(
                        "text",
                        ""
                    )
                    .replace("\n", " ")
                    .strip()
                )[:180]

                print(
                    f"{item['metadata'].get('file_name', 'Unknown')} "
                    f"=> list_score={item.get('_list_score', 0.0):.4f} "
                    f"hybrid={item.get('score', 0.0):.4f} "
                    f"| {preview}"
                )

            print("====================================\n")

        if useful_results:

            return useful_results

        return ranked

    def _focus_top_source_for_short_query(self, query, results):

        if not results:
            return results

        if not self._is_short_lookup_query(query):
            return results

        top_file = results[0]["metadata"].get(
            "file_name",
            "Unknown"
        )

        focused = [
            item for item in results
            if item["metadata"].get(
                "file_name",
                "Unknown"
            ) == top_file
        ]

        if DEBUG_RETRIEVAL:

            print("\n===== SHORT QUERY SOURCE FOCUS =====")
            print(f"Query    : {query}")
            print(f"Top File : {top_file}")
            print(f"Chunks   : {len(focused)}")
            print("===================================\n")

        if focused:
            return focused

        return results

    def _sentence_count(self, text):

        if not text:
            return 0

        return len(
            re.findall(
                r"[.!?]",
                text
            )
        )

    def _has_definition_signal(self, text):

        if not text:
            return False

        normalized = self._normalize_text(text)
        padded = f" {normalized} "

        # Important:
        # Do NOT treat plain "is", "was", or "are" as definition.
        # Example low-value chunk:
        # "It is located at Earl Bales Park..."
        #
        # That is not a useful definition for "Who is Jose Rizal?"
        definition_signals = [
            " was a ",
            " was an ",
            " is a ",
            " is an ",
            " was the ",
            " is the ",
            " refers to ",
            " known as ",
            " known for ",
            " served as ",
            " born ",
            " died ",
            " filipino ",
            " nationalist ",
            " writer ",
            " polymath ",
            " employee ",
            " employees ",
            " policy ",
            " procedure ",
            " standard ",
            " requirement ",
            " requirements ",
            " must ",
            " shall ",
        ]

        return any(
            signal in padded
            for signal in definition_signals
        )

    def _reference_noise_score(self, text):

        if not text:
            return 0

        normalized = self._normalize_text(text)
        padded = f" {normalized} "

        noise_terms = [
            " retrieved ",
            " archived ",
            " archive ",
            " http ",
            " https ",
            " www ",
            " isbn ",
            " issn ",
            " doi ",
            " references ",
            " external links ",
            " bibliography ",
            " further reading ",
            " web archive ",
            " wayback machine ",
            " google ",
            " abs cbn ",
            " gma news ",
            " inquirer ",
            " news ",
        ]

        score = 0

        for term in noise_terms:

            if term in padded:
                score += 1

        url_count = (
            text.lower().count("http")
            + text.lower().count("www.")
        )

        score += min(
            url_count,
            5
        )

        return score

    def _citation_marker_score(self, text):

        if not text:
            return 0

        clean_text = text.strip()

        citation_markers = re.findall(
            r"\[\d+\]",
            clean_text
        )

        score = min(
            len(citation_markers),
            8
        )

        # Stronger penalty only when chunk starts like a reference item.
        if re.match(r"^\[\d+\]", clean_text):
            score += 5

        return score

    def _query_match_score(self, query, text):

        if not query or not text:
            return 0.0

        query_tokens = self._normalize_text(query).split()
        text_normalized = self._normalize_text(text)

        if not query_tokens:
            return 0.0

        # Remove duplicate query tokens.
        # This prevents enriched queries like:
        # "jose rizal jose rizal biography jose rizal life..."
        # from over-boosting chunks just because they repeat the name.
        unique_tokens = []

        seen = set()

        for token in query_tokens:

            if token not in seen:

                seen.add(token)
                unique_tokens.append(token)

        score = 0.0
        first_part = text_normalized[:900]

        for token in unique_tokens:

            if len(token) <= 2:
                continue

            if token in first_part:
                score += 0.25

            elif token in text_normalized:
                score += 0.08

        # Cap query match score so it does not overpower
        # intro/definition quality.
        return min(
            score,
            1.25
        )

    def _meaningful_query_tokens(
        self,
        query
    ):

        """
        Extract useful tokens from the query for source/topic matching.

        Removes filler words so:
            "the treaty of paris in detail"
        becomes:
            ["treaty", "paris"]
        """

        normalized = self._normalize_text(
            query
        )

        stop_words = {
            "the",
            "a",
            "an",
            "of",
            "in",
            "on",
            "for",
            "to",
            "and",
            "or",
            "with",
            "about",
            "what",
            "who",
            "when",
            "where",
            "why",
            "how",
            "is",
            "are",
            "was",
            "were",
            "can",
            "you",
            "please",
            "explain",
            "describe",
            "tell",
            "me",
            "detail",
            "details",
            "definition",
            "overview",
            "description",
            "summary",
            "background",
            "important",
            "information",
            "key",
            "facts",
            "biography",
            "life",
            "known",
        }

        tokens = []

        seen = set()

        for token in normalized.split():

            if len(token) <= 2:

                continue

            if token in stop_words:

                continue

            if token not in seen:

                seen.add(
                    token
                )

                tokens.append(
                    token
                )

        return tokens

    def _file_topic_tokens(
        self,
        metadata
    ):

        """
        Extract meaningful tokens from the source file name.

        Examples:
            Treaty of Paris (1898) - Wikipedia.pdf
            -> ["treaty", "paris"]

            MISRA_FromInternet.pdf
            -> ["misra"]
        """

        if not metadata:

            return []

        file_name = (
            metadata.get(
                "file_name",
                ""
            )
            .strip()
        )

        if not file_name:

            return []

        # Remove extension.
        topic = re.sub(
            r"\.[^.]+$",
            "",
            file_name
        )

        # Remove common source suffixes.
        topic = re.sub(
            r"\s*-\s*Wikipedia$",
            "",
            topic,
            flags=re.IGNORECASE
        )

        topic = re.sub(
            r"[_\s-]*FromInternet$",
            "",
            topic,
            flags=re.IGNORECASE
        )

        # Remove year or extra text in parentheses.
        topic = re.sub(
            r"\([^)]*\)",
            " ",
            topic
        )

        normalized = self._normalize_text(
            topic
        )

        stop_words = {
            "the",
            "a",
            "an",
            "of",
            "in",
            "on",
            "for",
            "to",
            "and",
            "or",
            "with",
            "about",
            "wikipedia",
            "frominternet",
            "pdf",
            "docx",
            "txt",
        }

        tokens = []

        seen = set()

        for token in normalized.split():

            if len(token) <= 2:

                continue

            if token in stop_words:

                continue

            if token not in seen:

                seen.add(
                    token
                )

                tokens.append(
                    token
                )

        return tokens

    def _source_topic_score(
        self,
        query,
        item
    ):

        """
        Boost chunks whose source file name strongly matches the query topic.

        This helps exact-topic questions prefer the exact document.

        Example:
            Query: Treaty of Paris in detail
            File : Treaty of Paris (1898) - Wikipedia.pdf
            -> boosted

            File : Spanish-American War - Wikipedia.pdf
            -> not boosted
        """

        if not query or not item:

            return 0.0

        metadata = item.get(
            "metadata",
            {}
        )

        file_tokens = self._file_topic_tokens(
            metadata
        )

        query_tokens = self._meaningful_query_tokens(
            query
        )

        if not file_tokens or not query_tokens:

            return 0.0

        query_set = set(
            query_tokens
        )

        matched_tokens = [
            token for token in file_tokens
            if token in query_set
        ]

        # Single-token technical/source topic.
        # Example:
        # MISRA_FromInternet.pdf + query "misra"
        if (
            len(file_tokens) == 1
            and file_tokens[0] in query_set
            and len(file_tokens[0]) >= 4
        ):

            return 2.00

        # Multi-token exact topic.
        # Example:
        # Treaty of Paris -> treaty + paris
        if len(file_tokens) >= 2:

            coverage = (
                len(matched_tokens)
                / len(file_tokens)
            )

            if coverage >= 0.70:

                return 2.50

            if len(matched_tokens) >= 2:

                return 1.25

        return 0.0

    def _intro_definition_score(self, query, text):

        if not query or not text:
            return 0.0

        normalized_query = self._normalize_text(query)
        normalized_text = self._normalize_text(text)

        first_part = normalized_text[:1000]
        padded_first = f" {first_part} "

        score = 0.0

        if normalized_query in first_part:
            score += 0.40

        intro_patterns = [
            " was a ",
            " was an ",
            " is a ",
            " is an ",
            " was the ",
            " is the ",
            " filipino ",
            " employee ",
            " employees ",
            " policy ",
            " procedure ",
        ]

        if any(
            pattern in padded_first
            for pattern in intro_patterns
        ):
            score += 0.80

        return score

    def _is_identity_lookup_query(self, query):

        if not query:
            return False

        normalized = self._normalize_text(query)
        padded = f" {normalized} "

        identity_terms = [
            " biography ",
            " life ",
            " known for ",
            " important facts ",
            " who is ",
            " who was ",
            " sino si ",
            " sino ang ",
        ]

        return any(
            term in padded
            for term in identity_terms
        )


    def _identity_noise_score(self, text):

        if not text:
            return 0

        normalized = self._normalize_text(text)
        padded = f" {normalized} "

        # These terms are usually not useful for answering:
        # "Who is X?"
        identity_noise_terms = [
            " monument ",
            " statue ",
            " park ",
            " located at ",
            " unveiled ",
            " gift from ",
            " community of ",
            " city of ",
            " film ",
            " tv series ",
            " portrayed ",
            " actor ",
            " actress ",
            " plays ",
            " book ",
            " books ",
            " authored ",
            " google books ",
            " retrieved ",
            " archived ",
            " references ",
            " bibliography ",
            " external links ",
            " further reading ",
            " isbn ",
            " issn ",
            " doi ",
            " http ",
            " https ",
            " www ",
        ]

        score = 0

        for term in identity_noise_terms:

            if term in padded:
                score += 1

        return score


    def _strong_intro_score(self, query, text):

        if not query or not text:
            return 0.0

        normalized_query = self._normalize_text(query)
        normalized_text = self._normalize_text(text)

        first_part = normalized_text[:700]
        padded_first = f" {first_part} "

        score = 0.0

        query_tokens = normalized_query.split()

        meaningful_tokens = [
            token for token in query_tokens
            if len(token) > 2
            and token not in {
                "biography",
                "life",
                "known",
                "for",
                "background",
                "important",
                "facts",
                "overview",
                "summary",
                "description",
            }
        ]

        meaningful_tokens = list(
            dict.fromkeys(
                meaningful_tokens
            )
        )

        topic_match_count = 0

        for token in meaningful_tokens:

            if token in first_part:
                topic_match_count += 1

        if topic_match_count >= 2:
            score += 0.80

        strong_intro_patterns = [
            " was a ",
            " was an ",
            " is a ",
            " is an ",
            " was the ",
            " is the ",
            " was a filipino ",
            " was an filipino ",
            " filipino nationalist ",
            " writer and polymath ",
            " born ",
            " died ",
        ]

        if any(
            pattern in padded_first
            for pattern in strong_intro_patterns
        ):
            score += 1.20

        # Strong boost if this looks like the first paragraph
        # of an article/profile.
        if (
            topic_match_count >= 2
            and self._sentence_count(text) >= 1
            and len(text.split()) >= 40
        ):
            score += 0.50

        return score

    def _looks_like_low_value_chunk(self, text):

        if not text:
            return True

        clean_text = text.strip()
        words = clean_text.split()

        noise_score = self._reference_noise_score(clean_text)
        citation_score = self._citation_marker_score(clean_text)
        has_definition = self._has_definition_signal(clean_text)

        # Reject strong reference / URL chunks,
        # but do not reject real intro/definition paragraphs.
        if noise_score >= 8:
            return True

        # Reject chunks that start like citation/reference list items.
        if citation_score >= 8 and re.match(r"^\[\d+\]", clean_text):
            return True

        if len(words) < 25 and not has_definition:
            return True

        if (
            len(words) < 60
            and self._sentence_count(clean_text) == 0
            and not has_definition
        ):
            return True

        return False

    def _information_score(self, query, item):

        text = item.get(
            "text",
            ""
        )

        base_score = float(
            item.get(
                "score",
                0.0
            )
        )

        words = text.split()

        noise_score = self._reference_noise_score(text)
        citation_score = self._citation_marker_score(text)
        identity_noise_score = self._identity_noise_score(text)

        has_definition = self._has_definition_signal(text)

        is_identity_lookup = self._is_identity_lookup_query(
            query
        )

        info_score = base_score

        info_score += self._query_match_score(
            query,
            text
        )

        source_topic_score = self._source_topic_score(
            query,
            item
        )

        info_score += source_topic_score

        info_score += self._intro_definition_score(
            query,
            text
        )

        # Extra boost for "Who is X?" / biography-style queries.
        if is_identity_lookup:

            info_score += self._strong_intro_score(
                query,
                text
            )

        if len(words) >= 40:
            info_score += 0.10

        if len(words) >= 100:
            info_score += 0.15

        if has_definition:
            info_score += 0.45

        if self._sentence_count(text) >= 2:
            info_score += 0.10

        # Penalize URL/reference noise.
        info_score -= min(
            noise_score * 0.18,
            1.20
        )

        # Extra penalty for identity/person lookup when the chunk
        # is about monuments, parks, films, books, references, etc.
        if is_identity_lookup:

            info_score -= min(
                identity_noise_score * 0.35,
                2.10
            )

        # Citation markers are normal in Wikipedia intro paragraphs.
        # Penalize lightly when the chunk has a real definition.
        if has_definition:

            info_score -= min(
                citation_score * 0.05,
                0.40
            )

        else:

            info_score -= min(
                citation_score * 0.20,
                1.50
            )

        if self._looks_like_low_value_chunk(text):

            info_score -= 1.00

        item["_source_topic_score"] = source_topic_score
        item["_info_score"] = info_score
        item["_noise_score"] = noise_score
        item["_citation_score"] = citation_score
        item["_identity_noise_score"] = identity_noise_score
        item["_has_definition"] = has_definition

        return info_score

    def _prioritize_informative_chunks(self, query, results):

        if not results:
            return results

        ranked = sorted(
            results,
            key=lambda item: self._information_score(
                query,
                item
            ),
            reverse=True
        )

        if DEBUG_RETRIEVAL:

            print("\n===== INFORMATIVE CHUNK RANKING =====")

            for item in ranked[:10]:

                preview = (
                    item.get(
                        "text",
                        ""
                    )
                    .replace("\n", " ")
                    .strip()
                )[:150]

                print(
                    f"{item['metadata'].get('file_name', 'Unknown')} "
                    f"=> hybrid={item.get('score', 0.0):.4f} "
                    f"info={item.get('_info_score', 0.0):.4f} "
                    f"source={item.get('_source_topic_score', 0.0):.2f} "
                    f"noise={item.get('_noise_score', 0)} "
                    f"id_noise={item.get('_identity_noise_score', 0)} "
                    f"cite={item.get('_citation_score', 0)} "
                    f"def={item.get('_has_definition', False)} "
                    f"| {preview}"
                )

            print("====================================\n")

        return ranked

    def _remove_low_information_chunks(self, results):

        if not results:
            return results

        useful_results = [
            item for item in results
            if not self._looks_like_low_value_chunk(
                item.get(
                    "text",
                    ""
                )
            )
        ]

        # If everything is filtered out,
        # do not return empty. Keep the best sorted candidates.
        if useful_results:
            return useful_results

        return results

    def _remove_identity_noise_chunks(
        self,
        query,
        results
    ):

        if not results:
            return results

        if not self._is_identity_lookup_query(
            query
        ):
            return results

        useful_results = []

        for item in results:

            identity_noise_score = item.get(
                "_identity_noise_score",
                0
            )

            info_score = item.get(
                "_info_score",
                0.0
            )

            text = item.get(
                "text",
                ""
            )

            normalized_text = self._normalize_text(
                text
            )

            first_part = normalized_text[:900]
            padded_first = f" {first_part} "

            # These are strong person-biography signals.
            # Do NOT include generic "is a" / "was a" here,
            # because monument chunks can also contain that.
            true_biography_patterns = [
                " filipino nationalist ",
                " writer and polymath ",
                " ophthalmologist ",
                " national hero ",
                " propaganda movement ",
                " born ",
                " died ",
                " execution ",
                " spanish colonial period ",
                " political reforms ",
            ]

            has_true_biography_signal = any(
                pattern in padded_first
                for pattern in true_biography_patterns
            )

            # Strict rule:
            # If this chunk has high identity noise and does NOT have
            # strong biography signals, remove it immediately.
            if (
                identity_noise_score >= 6
                and not has_true_biography_signal
            ):

                continue

            # Remove weak medium-noise chunks.
            if (
                identity_noise_score >= 4
                and info_score < 1.50
                and not has_true_biography_signal
            ):

                continue

            useful_results.append(
                item
            )

        if useful_results:
            return useful_results

        # Safety fallback:
        # Return the best original candidate only.
        return results[:1]

    def _apply_diversity_filter(
        self,
        results,
        max_chunks_per_file=2
    ):

        filtered = []
        file_count = {}

        for item in results:

            file_name = item["metadata"].get(
                "file_name",
                "Unknown"
            )

            current_count = file_count.get(
                file_name,
                0
            )

            if current_count >= max_chunks_per_file:
                continue

            file_count[file_name] = current_count + 1
            filtered.append(item)

        return filtered

    def retrieve(self, query):

        if not query:

            return []

        is_list_mode = self._is_list_or_relationship_query(
            query
        )

        final_top_k = self._final_top_k_for_query(
            query
        )

        if is_list_mode:

            bm25_top_k = max(
                BM25_TOP_K,
                60
            )

            vector_top_k = max(
                VECTOR_TOP_K,
                60
            )

        else:

            bm25_top_k = BM25_TOP_K
            vector_top_k = VECTOR_TOP_K

        bm25_results = self.bm25.search(
            query,
            bm25_top_k
        )

        vector_results = self.chroma.search(
            query,
            vector_top_k
        )

        merged = self.hybrid.merge(
            vector_results,
            bm25_results
        )

        if DEBUG_RETRIEVAL:

            print("\n===== HYBRID RESULTS =====")

            for item in merged[:10]:

                print(
                    f"{item['metadata'].get('file_name', 'Unknown')} "
                    f"=> "
                    f"{item.get('score', 0.0):.4f}"
                )

            print("==========================\n")

            if is_list_mode:

                print("===== LIST MODE ENABLED =====")
                print(f"Query       : {query}")
                print(f"Final Top K : {final_top_k}")
                print("=============================\n")

        candidates = self._focus_top_source_for_short_query(
            query,
            merged
        )

        candidates = self._prioritize_informative_chunks(
            query,
            candidates
        )

        if is_list_mode:

            candidates = self._prioritize_list_relationship_chunks(
                query,
                candidates
            )

        else:

            candidates = self._remove_low_information_chunks(
                candidates
            )

            candidates = self._remove_identity_noise_chunks(
                query,
                candidates
            )

        # List questions need more candidates because answers
        # may be spread across several chunks.
        if is_list_mode:

            candidates = candidates[:20]

        else:

            candidates = candidates[:10]

        if (
            ENABLE_RERANKER
            and self.reranker
            and not is_list_mode
        ):

            ranked = self.reranker.rerank(
                query,
                candidates
            )

            ranked = ranked[:final_top_k]

            # ======================================
            # RERANKER CONFIDENCE GATE
            # ======================================
            # If the best reranker score is too low,
            # treat the query as not found.
            #
            # This prevents unrelated chunks from being
            # sent to the LLM for questions outside
            # the knowledge base.
            best_rerank_score = 0.0

            if ranked:

                best_rerank_score = float(
                    ranked[0].get(
                        "rerank_score",
                        0.0
                    )
                )

            if best_rerank_score < MIN_RETRIEVAL_SCORE:

                if DEBUG_RETRIEVAL:

                    print(
                        "\n===== RETRIEVAL CONFIDENCE GATE ====="
                    )

                    print(
                        f"Best rerank score : "
                        f"{best_rerank_score:.4f}"
                    )

                    print(
                        f"Minimum required  : "
                        f"{MIN_RETRIEVAL_SCORE:.4f}"
                    )

                    print(
                        "Result            : NO CONTEXT"
                    )

                    print(
                        "=====================================\n"
                    )

                return []

            # Keep only chunks that individually pass the
            # reranker confidence threshold.
            #
            # The previous logic checked only the best result.
            # That allowed low-confidence unrelated chunks to
            # enter the final context whenever the top result
            # passed the gate.
            ranked = [
                item
                for item in ranked
                if float(
                    item.get(
                        "rerank_score",
                        0.0
                    )
                ) >= MIN_RETRIEVAL_SCORE
            ]

            if DEBUG_RETRIEVAL:

                print(
                    "\n===== PER-RESULT CONFIDENCE FILTER ====="
                )

                print(
                    f"Minimum required : "
                    f"{MIN_RETRIEVAL_SCORE:.4f}"
                )

                print(
                    f"Accepted chunks  : "
                    f"{len(ranked)}"
                )

                for item in ranked:

                    print(
                        f"{item['metadata'].get('file_name', 'Unknown')} "
                        f"=> "
                        f"{item.get('rerank_score', 0.0):.4f}"
                    )

                print(
                    "========================================\n"
                )

                print("\n===== RERANKED RESULTS =====")

                for item in ranked[:10]:

                    print(
                        f"{item['metadata'].get('file_name', 'Unknown')} "
                        f"=> "
                        f"rerank={item.get('rerank_score', 0.0):.4f} "
                        f"hybrid={item.get('score', 0.0):.4f} "
                        f"info={item.get('_info_score', 0.0):.4f}"
                    )

                print("============================\n")

        else:

            ranked = candidates[:final_top_k]

            if DEBUG_RETRIEVAL:

                print("\n===== FINAL HYBRID RESULTS =====")

                for item in ranked:

                    print(
                        f"{item['metadata'].get('file_name', 'Unknown')} "
                        f"=> "
                        f"{item.get('score', 0.0):.4f} "
                        f"| info={item.get('_info_score', 0.0):.4f} "
                        f"noise={item.get('_noise_score', 0)} "
                        f"cite={item.get('_citation_score', 0)} "
                        f"def={item.get('_has_definition', False)}"
                    )

                print("===============================\n")

        if is_list_mode:

            ranked = self._expand_with_neighbor_chunks(
                ranked
            )

            ranked.sort(
                key=lambda item: (
                    item.get(
                        "metadata",
                        {}
                    ).get(
                        "file_name",
                        ""
                    ),
                    int(
                        item.get(
                            "metadata",
                            {}
                        ).get(
                            "chunk_id",
                            0
                        )
                    )
                )
            )

            max_chunks_per_file = final_top_k

        else:

            max_chunks_per_file = (
                FINAL_TOP_K
                if self._is_short_lookup_query(query)
                else 2
            )

        diversified = self._apply_diversity_filter(
            ranked,
            max_chunks_per_file=max_chunks_per_file
        )

        diversified = diversified[:final_top_k]

        if DEBUG_RETRIEVAL:

            print("\n===== DIVERSIFIED RESULTS =====")

            for item in diversified[:10]:

                print(
                    f"{item['metadata'].get('file_name', 'Unknown')} "
                    f"=> "
                    f"{item.get('rerank_score', item.get('score', 0.0)):.4f}"
                )

            print("===============================\n")

        return diversified
    
    def build_context(self, query):

        results = self.retrieve(
            query
        )

        results = results[
            :self._final_top_k_for_query(
                query
            )
        ]

        context = ""

        for index, item in enumerate(
            results,
            start=1
        ):

            context += (
                f"\n\n"
                f"===== DOCUMENT {index} =====\n"
                f"{item['text']}"
            )

        if DEBUG_RETRIEVAL:

            print("\n===== FINAL CONTEXT PREVIEW =====")

            for index, item in enumerate(
                results,
                start=1
            ):

                preview = (
                    item.get(
                        "text",
                        ""
                    )
                    .replace("\n", " ")
                    .strip()
                )[:300]

                print(
                    f"\nDOCUMENT {index}: "
                    f"{item['metadata'].get('file_name', 'Unknown')}"
                )

                print(preview)

            print("=================================\n")

        return context, results