from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from retrieval.retriever import CompanyRetriever
from services.misra_compliance import MisraComplianceMode

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "logs" / "pipeline_diagnostics"


def _reference_from_item(item) -> str:
    meta = item.get("metadata", {}) if isinstance(item, dict) else {}
    if not isinstance(meta, dict):
        return ""
    section_type = str(meta.get("section_type", "") or "").casefold()
    identifier = str(meta.get("directive_id", "") or meta.get("rule_id", "") or "").strip()
    if not identifier:
        return str(meta.get("section_title", "") or "").strip()
    return f"Directive {identifier}" if section_type == "directive" else f"Rule {identifier}"


def run_case(retriever: CompanyRetriever, question: str, expected_reference: str = "") -> dict:
    started = time.perf_counter()
    semantic = retriever.resolve_misra_rule_semantics(question)
    final_results = []
    context = ""
    path = "semantic_rule"

    if semantic.get("accepted"):
        records = MisraComplianceMode.load_authoritative_bm25_records()
        final_results = MisraComplianceMode.authoritative_semantic_reference_evidence(
            records,
            kind=str(semantic.get("kind", "")),
            identifier=str(semantic.get("identifier", "")),
            diagnostics=semantic,
        )
        context = MisraComplianceMode.build_grounded_context(final_results, question)
    else:
        path = "chunk_fallback"
        search_query = MisraComplianceMode.build_search_query(
            question=question,
            resolved_question=question,
        )
        context, final_results = retriever.build_context(
            search_query,
            intent_query=question,
            source_family="misra",
        )

    final_refs = list(dict.fromkeys(filter(None, (_reference_from_item(x) for x in final_results))))
    expected = str(expected_reference or "").strip()
    return {
        "question": question,
        "expected_reference": expected,
        "retrieval_path": path,
        "semantic_resolution": semantic,
        "final_references": final_refs,
        "final_chunks": [
            {
                "reference": _reference_from_item(item),
                "section_role": str((item.get("metadata", {}) or {}).get("section_role", "") or ""),
                "page_start": (item.get("metadata", {}) or {}).get("page_start"),
                "page_end": (item.get("metadata", {}) or {}).get("page_end"),
                "rerank_score": item.get("rerank_score"),
                "text": str(item.get("text", "") or ""),
            }
            for item in final_results
            if isinstance(item, dict)
        ],
        "verified_context": context,
        "retrieval_pass": (expected in final_refs) if expected else None,
        "seconds": round(time.perf_counter() - started, 4),
        "llm_answer_generation_called": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DocuBot retrieval-only MISRA diagnostic")
    parser.add_argument("--question", default="")
    parser.add_argument("--expected-reference", default="")
    parser.add_argument("--benchmark-json", default="")
    args = parser.parse_args()

    cases = []
    if args.benchmark_json:
        payload = json.loads(Path(args.benchmark_json).read_text(encoding="utf-8"))
        for case in payload.get("cases") or []:
            if isinstance(case, dict) and case.get("question"):
                cases.append((str(case["question"]), str(case.get("expected_reference", ""))))
    elif args.question:
        cases.append((args.question, args.expected_reference))
    else:
        parser.error("provide --question or --benchmark-json")

    retriever = CompanyRetriever()
    results = [run_case(retriever, q, expected) for q, expected in cases]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / f"retrieval_only_{stamp}.json"
    summary = {
        "mode": "retrieval_only",
        "created": datetime.now().isoformat(timespec="seconds"),
        "case_count": len(results),
        "passes": sum(1 for x in results if x.get("retrieval_pass") is True),
        "fails": sum(1 for x in results if x.get("retrieval_pass") is False),
        "cases": results,
    }
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out)
    for index, case in enumerate(results, 1):
        print(index, case["final_references"], f"{case['seconds']}s", case.get("retrieval_pass"))
    return 1 if any(x.get("retrieval_pass") is False for x in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
