import time
from pathlib import Path

from retrieval.retriever import (
    CompanyRetriever
)


def main():

    retriever = (
        CompanyRetriever()
    )

    while True:

        query = input(
            "\nQuestion: "
        )

        if query.lower() in [
            "exit",
            "quit"
        ]:
            break

        # =====================================
        # START TIMER
        # =====================================

        retrieval_start = (
            time.time()
        )

        results = (
            retriever.retrieve(
                query
            )
        )

        retrieval_time = (
            time.time()
            - retrieval_start
        )

        # =====================================
        # REPORT
        # =====================================

        report = []

        report.append(
            "=================================================="
        )

        report.append(
            "QUESTION"
        )

        report.append(
            "=================================================="
        )

        report.append("")
        report.append(query)
        report.append("")

        # =====================================
        # COUNTS
        # =====================================

        report.append(
            "=================================================="
        )

        report.append(
            "COUNTS"
        )

        report.append(
            "=================================================="
        )

        report.append("")

        report.append(
            f"Final retrieved docs : {len(results)}"
        )

        report.append("")

        # =====================================
        # TIMINGS
        # =====================================

        report.append(
            "=================================================="
        )

        report.append(
            "TIMINGS"
        )

        report.append(
            "=================================================="
        )

        report.append("")

        report.append(
            f"Retrieval : {retrieval_time:.2f} sec"
        )

        report.append("")

        # =====================================
        # RESULTS
        # =====================================

        report.append(
            "=================================================="
        )

        report.append(
            "FINAL RETRIEVED CONTEXT"
        )

        report.append(
            "=================================================="
        )

        report.append("")

        unique_files = set()

        best_score = None

        best_file = "N/A"

        for i, item in enumerate(
            results,
            start=1
        ):

            metadata = (
                item.get(
                    "metadata",
                    {}
                )
            )

            source = (
                metadata.get(
                    "file_name",
                    "Unknown"
                )
            )

            unique_files.add(
                source
            )

            rerank_score = (
                item.get(
                    "rerank_score",
                    0.0
                )
            )

            if (
                best_score is None
                or
                rerank_score > best_score
            ):

                best_score = (
                    rerank_score
                )

                best_file = (
                    source
                )

            preview = (
                item["text"]
                .replace(
                    "\n",
                    " "
                )
                [:700]
            )

            report.append(
                f"Result {i}"
            )

            report.append("")

            report.append(
                f"Source       : {source}"
            )

            report.append(
                f"Page         : "
                f"{metadata.get('page', 'N/A')}"
            )

            report.append(
                f"Chunk index  : "
                f"{metadata.get('chunk_id', 'N/A')}"
            )

            report.append(
                f"Chunk chars  : "
                f"{len(item['text'])}"
            )

            report.append("")

            report.append(
                f"Hybrid score : "
                f"{item.get('score', 0):.6f}"
            )

            report.append(
                f"Rerank score : "
                f"{rerank_score:.6f}"
            )

            report.append("")

            report.append(
                "Preview:"
            )

            report.append(
                preview
            )

            report.append("")
            report.append(
                "-" * 50
            )
            report.append("")

        # =====================================
        # SUMMARY
        # =====================================

        report.append(
            "=================================================="
        )

        report.append(
            "SUMMARY"
        )

        report.append(
            "=================================================="
        )

        report.append("")

        report.append(
            f"Query                : {query}"
        )

        report.append(
            f"Retrieved chunks     : {len(results)}"
        )

        report.append(
            f"Unique files         : {len(unique_files)}"
        )

        report.append(
            f"Top source           : {best_file}"
        )

        report.append(
            f"Best rerank score    : "
            f"{best_score:.6f}"
            if best_score is not None
            else
            "Best rerank score    : N/A"
        )

        report.append(
            f"Total retrieval time : "
            f"{retrieval_time:.2f} sec"
        )

        report.append("")

        final_report = (
            "\n".join(report)
        )

        print(
            "\n"
        )

        print(
            final_report
        )

        # =====================================
        # SAVE REPORT
        # =====================================

        output_file = Path(
            "retrieval_report.txt"
        )

        output_file.write_text(
            final_report,
            encoding="utf-8"
        )

        print(
            f"\nReport saved: {output_file}"
        )


if __name__ == "__main__":

    main()