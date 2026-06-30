from retrieval.retriever import (
    CompanyRetriever
)


class QueryService:

    def __init__(self):

        # Lazy-loaded retriever.
        # It will only be created when the
        # first search request is executed.
        self.retriever = None

    def _get_retriever(self):

        if self.retriever is None:

            print(
                "[RETRIEVER] Initializing..."
            )

            self.retriever = (
                CompanyRetriever()
            )

            print(
                "[RETRIEVER] Ready."
            )

        return self.retriever

    def retrieve_context(
        self,
        question: str
    ):

        retriever = (
            self._get_retriever()
        )

        context, results = (
            retriever.build_context(
                question
            )
        )

        return context, results
