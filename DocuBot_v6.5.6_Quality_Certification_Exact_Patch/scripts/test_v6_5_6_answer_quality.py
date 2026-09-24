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
from services.claim_grounding import validate_generated_claims

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "logs" / "v6_5_6_quality_certification"
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


def _pctl(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return round(ordered[index], 4)


def _run_case(client: OllamaClient, case: dict[str, Any]) -> dict[str, Any]:
    if not all(case.get(key) is True for key in ("retrieval_pass", "precision_pass", "top1_pass")):
        return {
            "question": case.get("question", ""),
            "concept_id": case.get("concept_id", ""),
            "expected_reference": case.get("expected_reference", ""),
            "blocked": True,
            "reason": "retrieval/precision/top-1 gate failed; answer generation intentionally blocked",
        }

    question = str(case.get("question", "") or "").strip()
    context = str(case.get("verified_context", "") or "").strip()
    expected = _canonical_ref(case.get("expected_reference", ""))
    expected_prefix = str(case.get("expected_prefix", "") or "").strip()
    if not question or not context:
        return {
            "question": question,
            "concept_id": case.get("concept_id", ""),
            "expected_reference": expected,
            "blocked": True,
            "reason": "verified question/context is empty",
        }

    prompt = (
        "Answer the USER QUESTION using ONLY the VERIFIED CONTEXT below. "
        "Do not introduce a Rule/Directive or factual claim that is absent from the context. "
        "When the context contains an applicable Rule/Directive reference, include that exact reference. "
        "For a yes/no compliance question, begin with Yes. or No. when the verified context determines the polarity. "
        "If the context does not determine the answer, say Needs more context. "
        "Be concise but state the source-grounded reason.\n\n"
        f"VERIFIED CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}"
    )

    started = time.perf_counter()
    answer = _answer_text(client.generate(prompt)).strip()
    seconds = time.perf_counter() - started

    refs = list(dict.fromkeys(_canonical_ref(match.group(0)) for match in REF_RE.finditer(answer)))
    wrong_refs = [ref for ref in refs if expected and ref != expected]
    reference_pass = bool(expected and expected in refs)
    polarity_pass = answer.casefold().startswith(expected_prefix.casefold()) if expected_prefix else True
    fallback = answer.casefold().startswith("needs more context")
    grounding = validate_generated_claims(answer=answer, context=context, question=question, results=())
    generation_pass = bool(reference_pass and polarity_pass and not wrong_refs and not fallback and grounding.ok)

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
        "reference_pass": reference_pass,
        "polarity_pass": polarity_pass,
        "claim_grounding_pass": bool(grounding.ok),
        "claim_grounding_reasons": list(grounding.reasons),
        "fallback": fallback,
        "generation_pass": generation_pass,
        "blocked": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DocuBot v6.5.6 answer-generation diagnostic using only verified retrieval context")
    parser.add_argument("--retrieval-json", required=True)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    payload = json.loads(Path(args.retrieval_json).read_text(encoding="utf-8"))
    cases = list(payload.get("cases") or [])
    if not cases:
        raise SystemExit("retrieval JSON contains no cases")

    client = OllamaClient(model_name=OLLAMA_FAST_MODEL)
    warmup: dict[str, Any] = {"executed": False}
    if not args.no_warmup:
        started = time.perf_counter()
        try:
            _answer_text(client.generate("Reply with exactly: OK"))
            warmup = {"executed": True, "ok": True, "seconds": round(time.perf_counter() - started, 4)}
        except Exception as error:
            warmup = {"executed": True, "ok": False, "seconds": round(time.perf_counter() - started, 4), "error": f"{type(error).__name__}: {error}"}

    results = [_run_case(client, case) for case in cases]
    executed = [item for item in results if not item.get("blocked")]
    blocked = [item for item in results if item.get("blocked")]
    passes = sum(item.get("generation_pass") is True for item in executed)
    grounding_passes = sum(item.get("claim_grounding_pass") is True for item in executed)
    latencies = [float(item.get("seconds") or 0.0) for item in executed]
    overall = bool(executed) and not blocked and passes == len(executed)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / f"v6.5.6_answer_diagnostic_{stamp}.json"
    result = {
        "version": "v6.5.6",
        "mode": "answer_generation_from_verified_context",
        "created": datetime.now().isoformat(timespec="seconds"),
        "overall": "PASS" if overall else "FAIL",
        "model": OLLAMA_FAST_MODEL,
        "warmup": warmup,
        "case_count": len(results),
        "executed": len(executed),
        "blocked": len(blocked),
        "generation_passes": passes,
        "claim_grounding_passes": grounding_passes,
        "latency_seconds": {
            "average": round(statistics.mean(latencies), 4) if latencies else None,
            "median_p50": round(statistics.median(latencies), 4) if latencies else None,
            "p95": _pctl(latencies, 0.95),
            "max": round(max(latencies), 4) if latencies else None,
        },
        "cases": results,
    }
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Overall: {result['overall']}")
    print(f"Generation: {passes}/{len(executed)}; blocked={len(blocked)}; grounding={grounding_passes}/{len(executed)}")
    print(f"Latency p50/p95/max: {result['latency_seconds']['median_p50']} / {result['latency_seconds']['p95']} / {result['latency_seconds']['max']} sec")
    print(out)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
