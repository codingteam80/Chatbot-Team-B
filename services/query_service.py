from retrieval.retriever import (
    CompanyRetriever
)
from qa.evidence_logger import evidence_logger


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

            evidence_logger.record_event(
                event_name="RETRIEVER INITIALIZATION",
                status="STARTED"
            )

            try:

                self.retriever = (
                    CompanyRetriever()
                )

                evidence_logger.record_event(
                    event_name="RETRIEVER INITIALIZATION",
                    status="READY"
                )

            except Exception as error:

                evidence_logger.record_error(
                    location="QueryService._get_retriever",
                    error=error
                )

                raise

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
