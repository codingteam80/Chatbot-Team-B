from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.retriever import CompanyRetriever
from services.misra_compliance import MisraComplianceMode

OUT_ROOT = ROOT / "logs" / "pipeline_diagnostics"
REF_RE = re.compile(r"\b(?:Rule|Directive)\s+\d+(?:\.\d+)+\b", re.I)


def _canonical_ref(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if text.casefold().startswith("rule "):
        return "Rule " + text.split(" ", 1)[1]
    if text.casefold().startswith("directive "):
        return "Directive " + text.split(" ", 1)[1]
    return text


def _reference_from_item(item: Any) -> str:
    meta = item.get("metadata", {}) if isinstance(item, dict) else {}
    if not isinstance(meta, dict):
        return ""
    section_type = str(meta.get("section_type", "") or "").casefold()
    identifier = str(meta.get("directive_id", "") or meta.get("rule_id", "") or "").strip()
    if not identifier:
        title = str(meta.get("section_title", "") or "").strip()
        match = REF_RE.search(title)
        return _canonical_ref(match.group(0)) if match else ""
    return f"Directive {identifier}" if section_type == "directive" else f"Rule {identifier}"


def _percentile95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return round(float(ordered[index]), 4)


def run_case(
    retriever: CompanyRetriever,
    question: str,
    expected_reference: str = "",
    *,
    concept_id: str = "",
    expected_prefix: str = "",
    variant: str = "",
) -> dict[str, Any]:
    started = time.perf_counter()
    semantic = retriever.resolve_misra_rule_semantics(question)
    final_results: list[dict[str, Any]] = []
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

    final_refs = list(
        dict.fromkeys(
            filter(None, (_reference_from_item(item) for item in final_results))
        )
    )
    expected = _canonical_ref(expected_reference)
    retrieval_pass = (expected in final_refs) if expected else None
    competing_refs = [ref for ref in final_refs if expected and ref != expected]
    precision_pass = (
        bool(retrieval_pass) and not competing_refs
        if expected
        else None
    )

    latency_profile = semantic.get("latency_profile") or {}
    decision = str(latency_profile.get("decision") or "")

    return {
        "question": question,
        "concept_id": str(concept_id or ""),
        "variant": str(variant or ""),
        "expected_reference": expected,
        "expected_prefix": str(expected_prefix or ""),
        "retrieval_path": path,
        "semantic_resolution": semantic,
        "semantic_latency_decision": decision,
        "multi_query_used": decision == "multi_query_used_for_ambiguous_single_query",
        "multi_query_skipped_single_query_proven": decision == "multi_query_skipped_single_query_proven",
        "final_references": final_refs,
        "competing_references": competing_refs,
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
        "retrieval_pass": retrieval_pass,
        "precision_pass": precision_pass,
        "seconds": round(time.perf_counter() - started, 4),
        "llm_answer_generation_called": False,
    }


def _load_cases(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.benchmark_json:
        payload = json.loads(Path(args.benchmark_json).read_text(encoding="utf-8"))
        cases = []
        for case in payload.get("cases") or []:
            if isinstance(case, dict) and case.get("question"):
                cases.append({
                    "question": str(case["question"]),
                    "expected_reference": str(case.get("expected_reference", "")),
                    "concept_id": str(case.get("concept_id", "")),
                    "expected_prefix": str(case.get("expected_prefix", "")),
                    "variant": str(case.get("variant", "")),
                })
        return cases

    if args.question:
        return [{
            "question": args.question,
            "expected_reference": args.expected_reference,
            "concept_id": "",
            "expected_prefix": "",
            "variant": "",
        }]

    raise SystemExit("provide --question or --benchmark-json")


def main() -> int:
    parser = argparse.ArgumentParser(description="DocuBot retrieval-only MISRA diagnostic")
    parser.add_argument("--question", default="")
    parser.add_argument("--expected-reference", default="")
    parser.add_argument("--benchmark-json", default="")
    args = parser.parse_args()

    cases = _load_cases(args)
    retriever = CompanyRetriever()
    results = [run_case(retriever, **case) for case in cases]

    concepts: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        concept_id = str(result.get("concept_id") or "")
        if concept_id:
            concepts.setdefault(concept_id, []).append(result)

    consistency: dict[str, bool] = {}
    for concept_id, items in concepts.items():
        if len(items) < 2:
            continue
        observed = [tuple(item.get("final_references") or []) for item in items]
        consistency[concept_id] = (
            all(item.get("retrieval_pass") is True and item.get("precision_pass") is True for item in items)
            and len(set(observed)) == 1
        )

    measured = [float(item["seconds"]) for item in results]
    retrieval_expected = [item for item in results if item.get("retrieval_pass") is not None]
    recall_passes = sum(item.get("retrieval_pass") is True for item in retrieval_expected)
    precision_passes = sum(item.get("precision_pass") is True for item in retrieval_expected)
    consistency_passes = sum(consistency.values())
    multi_query_used = sum(bool(item.get("multi_query_used")) for item in results)
    multi_query_skipped = sum(bool(item.get("multi_query_skipped_single_query_proven")) for item in results)

    overall = (
        len(retrieval_expected) == len(results)
        and recall_passes == len(results)
        and precision_passes == len(results)
        and (not consistency or consistency_passes == len(consistency))
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / f"retrieval_only_{stamp}.json"
    summary = {
        "version": "v6.5.5",
        "mode": "retrieval_only",
        "created": datetime.now().isoformat(timespec="seconds"),
        "overall": "PASS" if overall else "FAIL",
        "case_count": len(results),
        "recall_passes": recall_passes,
        "precision_passes": precision_passes,
        "concept_consistency_passes": consistency_passes,
        "concept_consistency_total": len(consistency),
        "concept_consistency": consistency,
        "multi_query_used_count": multi_query_used,
        "multi_query_skipped_single_query_proven_count": multi_query_skipped,
        "latency_seconds": {
            "average": round(statistics.mean(measured), 4) if measured else None,
            "median": round(statistics.median(measured), 4) if measured else None,
            "p95": _percentile95(measured),
            "max": round(max(measured), 4) if measured else None,
        },
        "llm_answer_generation_called": False,
        "cases": results,
    }
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Overall: {summary['overall']}")
    print(f"Recall: {recall_passes}/{len(results)}")
    print(f"Precision: {precision_passes}/{len(results)}")
    print(f"Consistency: {consistency_passes}/{len(consistency)}")
    print(f"MultiQuery used/skipped-proven: {multi_query_used}/{multi_query_skipped}")
    print(
        "Latency avg/median/p95/max: "
        f"{summary['latency_seconds']['average']} / "
        f"{summary['latency_seconds']['median']} / "
        f"{summary['latency_seconds']['p95']} / "
        f"{summary['latency_seconds']['max']} sec"
    )
    print(out)
    for index, case in enumerate(results, 1):
        print(
            index,
            case["final_references"],
            f"{case['seconds']}s",
            case.get("semantic_latency_decision"),
            case.get("retrieval_pass"),
            case.get("precision_pass"),
        )
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
