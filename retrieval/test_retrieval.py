from retrieval.hybrid_search import HybridRetriever


# SAMPLE DOCUMENTS (temporary test)
documents = [
    "Python is a programming language used for AI and web development.",
    "ChromaDB is a vector database for storing embeddings.",
    "BM25 is a keyword-based retrieval algorithm.",
    "RAG combines retrieval and generation for better AI answers.",
    "LangChain is a framework for building LLM applications."
]


def main():

    query = "What is retrieval in AI?"

    retriever = HybridRetriever(documents)

    results = retriever.retrieve(query)

    print("\n=== FINAL RESULTS ===\n")

    for i, r in enumerate(results, 1):
        print(f"{i}. SCORE: {r['score']:.4f}")
        print(f"   TEXT: {r['content']}\n")


if __name__ == "__main__":
    main()