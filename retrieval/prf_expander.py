from collections import Counter
import re


class PRFExpander:

    def __init__(self):

        # Store common words excluded from expansion.
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

        # Combine text from top BM25 results.
        text = " ".join(

            item["text"]

            for item in bm25_results[:3]

        )

        # Extract candidate keywords from retrieved text.
        words = re.findall(

            r"\b[a-zA-Z]{4,}\b",

            text.lower()

        )

        # Remove common stopwords.
        words = [

            word

            for word in words

            if word not in self.stopwords

        ]

        # Find the most frequent keywords.
        most_common = (

            Counter(words)

            .most_common(top_terms)

        )

        # Extract keyword text only.
        terms = [

            word

            for word, _ in most_common

        ]

        # Append keywords to the original query.
        expanded_query = (

            query
            + " "
            + " ".join(terms)

        )

        # Return the expanded query.
        return expanded_query