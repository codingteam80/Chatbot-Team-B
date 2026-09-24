from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from retrieval.retriever import CompanyRetriever
from services.misra_compliance import MisraComplianceMode

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "logs" / "v6_5_6_quality_certification"
REF_RE = re.compile(r"\b(?:Rule|Directive)\s+\d+(?:\.\d+)+\b", re.I)
WARMUP_QUERY = "Which MISRA requirement applies when a function calls itself indirectly through another function?"


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


def _pctl(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return round(ordered[index], 4)


def _load_cases(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[dict[str, str]] = []
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


def run_case(retriever: CompanyRetriever, case: dict[str, str]) -> dict[str, Any]:
    question = str(case.get("question", "") or "").strip()
    expected = _canonical_ref(case.get("expected_reference", ""))
    started = time.perf_counter()

    semantic_started = time.perf_counter()
    semantic = retriever.resolve_misra_rule_semantics(question)
    semantic_seconds = time.perf_counter() - semantic_started

    context_started = time.perf_counter()
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
    context_seconds = time.perf_counter() - context_started
    total_seconds = time.perf_counter() - started

    final_refs = list(dict.fromkeys(filter(None, (_reference_from_item(item) for item in final_results))))
    top_ref = final_refs[0] if final_refs else ""
    rank = final_refs.index(expected) + 1 if expected and expected in final_refs else None
    recall_at_3 = 1.0 if expected and expected in final_refs[:3] else 0.0
    precision_at_3 = (1.0 / len(final_refs[:3])) if expected and expected in final_refs[:3] and final_refs[:3] else 0.0
    competing_refs = [ref for ref in final_refs if expected and ref != expected]
    retrieval_pass = bool(expected and expected in final_refs)
    top1_pass = bool(expected and top_ref == expected)
    precision_pass = bool(retrieval_pass and not competing_refs)

    latency_profile = dict(semantic.get("latency_profile") or {})
    decision = str(latency_profile.get("decision") or "")
    alternatives = list(semantic.get("multi_query_queries") or [])
    alternative_count = max(0, len(alternatives) - 1) if alternatives else 0
    mq_used = decision == "multi_query_used_for_ambiguous_single_query"
    mq_skip = decision == "multi_query_skipped_single_query_proven"
    mq_contract_pass = (alternative_count == 2) if mq_used else True

    return {
        "question": question,
        "concept_id": str(case.get("concept_id", "")),
        "variant": str(case.get("variant", "")),
        "expected_reference": expected,
        "expected_prefix": str(case.get("expected_prefix", "")),
        "retrieval_path": path,
        "semantic_resolution": semantic,
        "semantic_latency_decision": decision,
        "multi_query_used": mq_used,
        "multi_query_skipped_single_query_proven": mq_skip,
        "multi_query_alternative_count": alternative_count,
        "multi_query_contract_pass": mq_contract_pass,
        "final_references": final_refs,
        "top_reference": top_ref,
        "expected_rank": rank,
        "competing_references": competing_refs,
        "recall_at_3": recall_at_3,
        "precision_at_3": round(precision_at_3, 4),
        "retrieval_pass": retrieval_pass,
        "top1_pass": top1_pass,
        "precision_pass": precision_pass,
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
        "latency_seconds": {
            "semantic_resolution": round(semantic_seconds, 4),
            "resolver_ready": latency_profile.get("resolver_ready_seconds"),
            "single_query": latency_profile.get("single_query_seconds"),
            "multi_query_generation": latency_profile.get("multi_query_generation_seconds"),
            "widened_resolution": latency_profile.get("widened_resolution_seconds"),
            "context_build": round(context_seconds, 4),
            "total_retrieval": round(total_seconds, 4),
        },
        "seconds": round(total_seconds, 4),
        "llm_answer_generation_called": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DocuBot v6.5.6 retrieval correctness / recall / precision / semantic consistency diagnostic")
    parser.add_argument("--benchmark-json", required=True)
    parser.add_argument("--no-warmup", action="store_true", help="include cold-start initialization in measured cases")
    args = parser.parse_args()

    cases = _load_cases(Path(args.benchmark_json))
    if not cases:
        raise SystemExit("benchmark JSON contains no cases")

    retriever = CompanyRetriever()
    warmup: dict[str, Any] = {"executed": False}
    if not args.no_warmup:
        started = time.perf_counter()
        try:
            retriever.resolve_misra_rule_semantics(WARMUP_QUERY)
            warmup = {"executed": True, "ok": True, "seconds": round(time.perf_counter() - started, 4), "question": WARMUP_QUERY}
        except Exception as error:
            warmup = {"executed": True, "ok": False, "seconds": round(time.perf_counter() - started, 4), "question": WARMUP_QUERY, "error": f"{type(error).__name__}: {error}"}

    results = [run_case(retriever, case) for case in cases]

    concepts: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        concept_id = str(result.get("concept_id") or "")
        if concept_id:
            concepts.setdefault(concept_id, []).append(result)

    consistency: dict[str, dict[str, Any]] = {}
    for concept_id, items in concepts.items():
        if len(items) < 2:
            continue
        expected = str(items[0].get("expected_reference") or "")
        top_refs = [str(item.get("top_reference") or "") for item in items]
        full_sets = [tuple(item.get("final_references") or []) for item in items]
        pair_pass = (
            all(item.get("retrieval_pass") is True for item in items)
            and all(item.get("precision_pass") is True for item in items)
            and all(item.get("top1_pass") is True for item in items)
            and len(set(top_refs)) == 1
            and top_refs[0] == expected
        )
        consistency[concept_id] = {
            "pass": bool(pair_pass),
            "expected_reference": expected,
            "top_references": top_refs,
            "top_reference_consistent": len(set(top_refs)) == 1,
            "full_reference_set_consistent": len(set(full_sets)) == 1,
        }

    latencies = [float(item.get("seconds") or 0.0) for item in results]
    recall_passes = sum(item.get("retrieval_pass") is True for item in results)
    precision_passes = sum(item.get("precision_pass") is True for item in results)
    top1_passes = sum(item.get("top1_pass") is True for item in results)
    mq_contract_passes = sum(item.get("multi_query_contract_pass") is True for item in results)
    consistency_passes = sum(bool(item.get("pass")) for item in consistency.values())
    multi_query_used = sum(bool(item.get("multi_query_used")) for item in results)
    multi_query_skipped = sum(bool(item.get("multi_query_skipped_single_query_proven")) for item in results)

    overall = bool(results) and all([
        recall_passes == len(results),
        precision_passes == len(results),
        top1_passes == len(results),
        mq_contract_passes == len(results),
        (not consistency or consistency_passes == len(consistency)),
    ])

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / f"v6.5.6_retrieval_benchmark_{stamp}.json"
    payload = {
        "version": "v6.5.6",
        "mode": "retrieval_quality_only",
        "created": datetime.now().isoformat(timespec="seconds"),
        "overall": "PASS" if overall else "FAIL",
        "warmup": warmup,
        "case_count": len(results),
        "recall_passes": recall_passes,
        "precision_passes": precision_passes,
        "top1_passes": top1_passes,
        "concept_consistency_passes": consistency_passes,
        "concept_consistency_total": len(consistency),
        "concept_consistency": consistency,
        "multi_query_used_count": multi_query_used,
        "multi_query_skipped_single_query_proven_count": multi_query_skipped,
        "multi_query_exactly_two_alternatives_passes": mq_contract_passes,
        "metrics": {
            "recall_at_3": round(sum(float(i.get("recall_at_3") or 0.0) for i in results) / len(results), 4),
            "mean_precision_at_3": round(sum(float(i.get("precision_at_3") or 0.0) for i in results) / len(results), 4),
        },
        "latency_seconds": {
            "average": round(statistics.mean(latencies), 4),
            "median_p50": round(statistics.median(latencies), 4),
            "p95": _pctl(latencies, 0.95),
            "max": round(max(latencies), 4),
        },
        "llm_answer_generation_called": False,
        "cases": results,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Overall: {payload['overall']}")
    print(f"Recall: {recall_passes}/{len(results)}")
    print(f"Precision: {precision_passes}/{len(results)}")
    print(f"Top-1: {top1_passes}/{len(results)}")
    print(f"Semantic consistency: {consistency_passes}/{len(consistency)}")
    print(f"MultiQuery used / single-query early-accept: {multi_query_used}/{multi_query_skipped}")
    print(f"Latency p50/p95/max: {payload['latency_seconds']['median_p50']} / {payload['latency_seconds']['p95']} / {payload['latency_seconds']['max']} sec")
    print(out)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
