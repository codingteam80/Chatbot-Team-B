from collections import Counter
import re


class PRFExpander:

    def __init__(self):

        # Common words to ignore
        self.stopwords = {

            "the",
            "a",
            "an",
            "and",
            "or",
            "of",
            "to",
            "for",
            "in",
            "on",
            "with",
            "is",
            "are",
            "was",
            "were",
            "this",
            "that",
            "it"
        }

    def expand(
        self,
        query,
        bm25_results,
        top_terms=5
    ):

        # Collect text from top BM25 chunks
        text = " ".join(

            item["text"]

            for item in bm25_results[:3]

        )

        # Extract words
        words = re.findall(

            r"\b[a-zA-Z]{4,}\b",

            text.lower()

        )

        # Remove stopwords
        words = [

            word

            for word in words

            if word not in self.stopwords

        ]

        # Get most frequent terms
        most_common = (

            Counter(words)

            .most_common(top_terms)

        )

        terms = [

            word

            for word, _ in most_common

        ]

        # Expand original query
        expanded_query = (

            query
            + " "
            + " ".join(terms)

        )

        return expanded_query