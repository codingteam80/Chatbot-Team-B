from retrieval.retriever import (
    CompanyRetriever
)
import time

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
            started = time.perf_counter()

            try:

                self.retriever = (
                    CompanyRetriever()
                )

                evidence_logger.record_event(
                    event_name="RETRIEVER INITIALIZATION",
                    status="READY",
                    details={
                        "seconds": round(time.perf_counter() - started, 4),
                        "note": (
                            "Heavy vector/reranker components remain lazy until "
                            "a non-structured query needs them."
                        ),
                    },
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
        question: str,
        intent_question: str | None = None,
        source_family: str | None = None,
        final_top_k_override: int | None = None,
    ):

        retriever = (
            self._get_retriever()
        )

        context, results = (
            retriever.build_context(
                question,
                intent_query=intent_question,
                source_family=source_family,
                final_top_k_override=final_top_k_override,
            )
        )

        return context, results
