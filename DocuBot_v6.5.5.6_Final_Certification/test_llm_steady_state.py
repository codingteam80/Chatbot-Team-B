from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

QA_DIR = Path(__file__).resolve().parent
ROOT = QA_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import OLLAMA_FAST_MODEL
from llm.ollama_client import OllamaClient
from runtime.prewarm import prewarm_fast_llm_sync

OUT_ROOT = ROOT / "logs" / "pipeline_diagnostics"
REF_RE = re.compile(r"\b(?:Rule|Directive)\s+\d+(?:\.\d+)+\b", re.I)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _canonical_ref(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if text.casefold().startswith("rule "):
        return "Rule " + text.split(" ", 1)[1]
    if text.casefold().startswith("directive "):
        return "Directive " + text.split(" ", 1)[1]
    return text


def _answer_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, tuple):
        return " ".join(str(part) for part in value if part is not None)
    return str(value or "")


def _run_case(client: OllamaClient, case: dict[str, Any]) -> dict[str, Any]:
    if case.get("retrieval_pass") is False or case.get("precision_pass") is False:
        return {
            "question": case.get("question", ""),
            "concept_id": case.get("concept_id", ""),
            "blocked": True,
            "reason": "retrieval gate failed; LLM generation intentionally not executed",
        }

    question = str(case.get("question", "") or "").strip()
    context = str(case.get("verified_context", "") or "").strip()
    if not question or not context:
        return {
            "question": question,
            "concept_id": case.get("concept_id", ""),
            "blocked": True,
            "reason": "verified question/context is empty",
        }

    expected = _canonical_ref(case.get("expected_reference", ""))
    expected_prefix = str(case.get("expected_prefix", "") or "").strip()
    prompt = (
        "Answer the USER QUESTION using ONLY the VERIFIED CONTEXT below. "
        "Do not introduce a Rule/Directive or factual claim that is absent from the context. "
        "When the context contains an applicable Rule/Directive reference, include that exact reference in the answer. "
        "For a yes/no compliance question, begin with Yes. or No. when the verified context determines the polarity. "
        "Use the same natural language as the USER QUESTION; do not switch to an unrelated language or script. "
        "If the context does not determine the answer, say Needs more context.\n\n"
        f"VERIFIED CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}"
    )

    started = time.perf_counter()
    answer = _answer_text(client.generate(prompt)).strip()
    seconds = time.perf_counter() - started
    refs = list(dict.fromkeys(_canonical_ref(match.group(0)) for match in REF_RE.finditer(answer)))
    wrong_refs = [ref for ref in refs if expected and ref != expected]
    ref_pass = (expected in refs) if expected else True
    prefix_pass = answer.casefold().startswith(expected_prefix.casefold()) if expected_prefix else True
    fallback = answer.casefold().startswith("needs more context")
    unexpected_cjk = bool(CJK_RE.search(answer)) and not bool(CJK_RE.search(question))
    grounded_pass = ref_pass and prefix_pass and not wrong_refs and not fallback

    return {
        "question": question,
        "concept_id": case.get("concept_id", ""),
        "variant": case.get("variant", ""),
        "expected_reference": expected,
        "expected_prefix": expected_prefix,
        "retrieved_references": case.get("final_references", []),
        "latency_class": case.get("latency_class", "normal_single_query"),
        "multi_query_used": bool(case.get("multi_query_used")),
        "model": OLLAMA_FAST_MODEL,
        "seconds": round(seconds, 4),
        "answer": answer,
        "answer_references": refs,
        "wrong_references": wrong_refs,
        "reference_pass": ref_pass,
        "polarity_pass": prefix_pass,
        "fallback": fallback,
        "generation_pass": grounded_pass,
        "language_script_warning": unexpected_cjk,
        "blocked": False,
    }


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "average": round(statistics.mean(values), 4) if values else None,
        "median": round(statistics.median(values), 4) if values else None,
        "max": round(max(values), 4) if values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DocuBot steady-state LLM diagnostic over verified retrieval context")
    parser.add_argument("--retrieval-json", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.retrieval_json).read_text(encoding="utf-8"))
    cases = list(payload.get("cases") or [])
    if not cases:
        raise SystemExit("retrieval JSON contains no cases")

    # Production startup prewarms the fast Ollama model. Exclude that startup
    # cost from normal steady-state question latency and record it separately.
    prewarm = prewarm_fast_llm_sync()
    if not prewarm.get("ok"):
        print("[FAIL] Fast LLM prewarm failed:", prewarm)
        return 2

    client = OllamaClient(model_name=OLLAMA_FAST_MODEL)
    results = [_run_case(client, case) for case in cases]
    executed = [item for item in results if not item.get("blocked")]
    blocked = [item for item in results if item.get("blocked")]
    passes = sum(item.get("generation_pass") is True for item in executed)
    warnings = sum(bool(item.get("language_script_warning")) for item in executed)
    latencies = [float(item.get("seconds") or 0.0) for item in executed]
    normal_latencies = [float(item.get("seconds") or 0.0) for item in executed if not item.get("multi_query_used")]
    mq_latencies = [float(item.get("seconds") or 0.0) for item in executed if item.get("multi_query_used")]
    overall = bool(executed) and not blocked and passes == len(executed)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / f"llm_steady_state_{stamp}.json"
    result = {
        "version": "v6.5.5",
        "qa_harness_version": "v6.5.5.6-final-certification",
        "mode": "llm_from_verified_context_after_production_equivalent_prewarm",
        "created": datetime.now().isoformat(timespec="seconds"),
        "overall": "PASS" if overall else "FAIL",
        "model": OLLAMA_FAST_MODEL,
        "prewarm": prewarm,
        "case_count": len(results),
        "executed": len(executed),
        "blocked": len(blocked),
        "generation_passes": passes,
        "language_script_warnings": warnings,
        "latency_seconds": {
            "all": _latency_summary(latencies),
            "normal_single_query": _latency_summary(normal_latencies),
            "ambiguous_multi_query": _latency_summary(mq_latencies),
        },
        "cases": results,
    }
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Prewarm: {prewarm.get('seconds')} sec (excluded from steady-state timers)")
    print(f"Overall: {result['overall']}")
    print(f"Generation: {passes}/{len(executed)}; blocked={len(blocked)}; language-script-warnings={warnings}")
    print("Normal generation latency:", result["latency_seconds"]["normal_single_query"])
    print("Ambiguous MQ generation latency:", result["latency_seconds"]["ambiguous_multi_query"])
    print(out)
    for index, item in enumerate(results, 1):
        if item.get("blocked"):
            print(index, "BLOCKED", item.get("reason"))
        else:
            print(index, item.get("latency_class"), item.get("expected_reference"), f"{item.get('seconds')}s", item.get("generation_pass"), item.get("answer"))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
