from collections import Counter


class PRFExpander:
    """
    Pseudo Relevance Feedback (query expansion).
    """

    def __init__(self, top_n_terms: int = 5):
        self.top_n_terms = top_n_terms

    def expand_query(self, query: str, top_docs: list):

        words = query.split()

        tokens = []
        for doc in top_docs:
            tokens.extend(doc["content"].split())

        common = Counter(tokens).most_common(self.top_n_terms)
        expansion_terms = [w for w, _ in common]

        return " ".join(words + expansion_terms)