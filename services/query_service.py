from retrieval.retriever import (
    CompanyRetriever
)


class QueryService:

    def __init__(self):

        # Main retrieval pipeline
        self.retriever = (
            CompanyRetriever()
        )

    def retrieve_context(
        self,
        question: str
    ):

        # Retrieve best chunks and build context
        context, results = (
            self.retriever.build_context(
                question
            )
        )

        # Return context and retrieval results
        return context, results