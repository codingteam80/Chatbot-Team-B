from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import OLLAMA_FAST_MODEL
from llm.ollama_client import OllamaClient

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "logs" / "pipeline_diagnostics"
REF_RE = re.compile(r"\b(?:Rule|Directive)\s+\d+(?:\.\d+)+\b", re.I)


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
            "expected_reference": case.get("expected_reference", ""),
            "expected_prefix": case.get("expected_prefix", ""),
            "blocked": True,
            "reason": "retrieval gate failed; LLM generation intentionally not executed",
        }

    question = str(case.get("question", "") or "").strip()
    context = str(case.get("verified_context", "") or "").strip()
    if not question or not context:
        return {
            "question": question,
            "concept_id": case.get("concept_id", ""),
            "expected_reference": case.get("expected_reference", ""),
            "expected_prefix": case.get("expected_prefix", ""),
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
        "If the context does not determine the answer, say Needs more context.\n\n"
        f"VERIFIED CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}"
    )

    started = time.perf_counter()
    answer = _answer_text(client.generate(prompt)).strip()
    seconds = time.perf_counter() - started
    refs = list(dict.fromkeys(_canonical_ref(match.group(0)) for match in REF_RE.finditer(answer)))
    wrong_refs = [ref for ref in refs if expected and ref != expected]
    ref_pass = (expected in refs) if expected else True
    prefix_pass = (
        answer.casefold().startswith(expected_prefix.casefold())
        if expected_prefix
        else True
    )
    fallback = answer.casefold().startswith("needs more context")
    grounded_pass = ref_pass and prefix_pass and not wrong_refs and not fallback

    return {
        "question": question,
        "concept_id": case.get("concept_id", ""),
        "variant": case.get("variant", ""),
        "expected_reference": expected,
        "expected_prefix": expected_prefix,
        "retrieved_references": case.get("final_references", []),
        "model": OLLAMA_FAST_MODEL,
        "seconds": round(seconds, 4),
        "answer": answer,
        "answer_references": refs,
        "wrong_references": wrong_refs,
        "reference_pass": ref_pass,
        "polarity_pass": prefix_pass,
        "fallback": fallback,
        "generation_pass": grounded_pass,
        "blocked": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DocuBot LLM-only diagnostic over pre-verified retrieval context")
    parser.add_argument("--retrieval-json", required=True)
    parser.add_argument("--case", type=int, default=0, help="1-based case number; 0 means all retrieval-pass cases")
    args = parser.parse_args()

    payload = json.loads(Path(args.retrieval_json).read_text(encoding="utf-8"))
    cases = list(payload.get("cases") or [])
    if not cases:
        raise SystemExit("retrieval JSON contains no cases")

    if args.case:
        index = max(1, int(args.case)) - 1
        if index >= len(cases):
            raise SystemExit("case index is outside retrieval JSON")
        selected = [cases[index]]
    else:
        selected = cases

    client = OllamaClient(model_name=OLLAMA_FAST_MODEL)
    results = [_run_case(client, case) for case in selected]
    executed = [item for item in results if not item.get("blocked")]
    blocked = [item for item in results if item.get("blocked")]
    passes = sum(item.get("generation_pass") is True for item in executed)
    latencies = [float(item.get("seconds") or 0.0) for item in executed]
    overall = bool(executed) and not blocked and passes == len(executed)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / f"llm_from_verified_context_{stamp}.json"
    result = {
        "version": "v6.5.5",
        "mode": "llm_only_from_verified_context",
        "created": datetime.now().isoformat(timespec="seconds"),
        "overall": "PASS" if overall else "FAIL",
        "model": OLLAMA_FAST_MODEL,
        "case_count": len(results),
        "executed": len(executed),
        "blocked": len(blocked),
        "generation_passes": passes,
        "latency_seconds": {
            "average": round(statistics.mean(latencies), 4) if latencies else None,
            "median": round(statistics.median(latencies), 4) if latencies else None,
            "max": round(max(latencies), 4) if latencies else None,
        },
        "cases": results,
    }
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Overall: {result['overall']}")
    print(f"Generation: {passes}/{len(executed)}; blocked={len(blocked)}")
    print(
        "Latency avg/median/max: "
        f"{result['latency_seconds']['average']} / "
        f"{result['latency_seconds']['median']} / "
        f"{result['latency_seconds']['max']} sec"
    )
    print(out)
    for index, item in enumerate(results, 1):
        if item.get("blocked"):
            print(index, "BLOCKED", item.get("reason"))
        else:
            print(
                index,
                item.get("expected_reference"),
                f"{item.get('seconds')}s",
                item.get("generation_pass"),
                item.get("answer"),
            )
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
