from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import OLLAMA_FAST_MODEL
from llm.ollama_client import OllamaClient
from runtime.prewarm import prewarm_fast_llm_sync
from services.verified_answer_contract import enforce_verified_answer_contract

OUT_ROOT = ROOT / "logs" / "v6_5_5_7_closure"
REF_RE = re.compile(r"\b(?:Rule|Directive)\s+\d+(?:\.\d+)+\b", re.I)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def _latest_retrieval_json() -> Path | None:
    folder = ROOT / "logs" / "pipeline_diagnostics"
    if not folder.exists():
        return None
    candidates = sorted(folder.glob("retrieval_steady_state_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(folder.glob("retrieval_only_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _canonical_ref(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    match = REF_RE.search(text)
    if not match:
        return ""
    raw = match.group(0)
    if raw.casefold().startswith("rule "):
        return "Rule " + raw.split(" ", 1)[1]
    return "Directive " + raw.split(" ", 1)[1]


def _synthetic_results(references: list[str]) -> list[dict[str, Any]]:
    results = []
    for raw in references:
        ref = _canonical_ref(raw)
        if not ref:
            continue
        kind, identifier = ref.split(" ", 1)
        if kind == "Rule":
            metadata = {"section_type": "rule", "rule_id": identifier}
        else:
            metadata = {"section_type": "directive", "directive_id": identifier}
        results.append({"metadata": metadata})
    return results


def _answer_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, tuple):
        return " ".join(str(part) for part in value if part is not None)
    return str(value or "")


def main() -> int:
    parser = argparse.ArgumentParser(description="DocuBot v6.5.5.7 live verified-answer contract test")
    parser.add_argument("--retrieval-json", default="")
    args = parser.parse_args()

    retrieval_path = Path(args.retrieval_json).resolve() if args.retrieval_json else _latest_retrieval_json()
    if not retrieval_path or not retrieval_path.exists():
        print("[FAIL] No prior verified retrieval diagnostic was found.")
        print("Run the v6.5.5.6 final certification retrieval stage first.")
        return 2

    payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
    cases = [case for case in (payload.get("cases") or []) if case.get("retrieval_pass") is not False and case.get("precision_pass") is not False]
    if not cases:
        print("[FAIL] Retrieval diagnostic contains no verified-pass cases.")
        return 2

    prewarm = prewarm_fast_llm_sync()
    if not prewarm.get("ok"):
        print("[FAIL] Fast LLM prewarm failed:", prewarm)
        return 2
    client = OllamaClient(model_name=OLLAMA_FAST_MODEL)

    results = []
    for index, case in enumerate(cases, 1):
        question = str(case.get("question", "") or "").strip()
        context = str(case.get("verified_context", "") or "").strip()
        expected = _canonical_ref(case.get("expected_reference", ""))
        expected_prefix = str(case.get("expected_prefix", "") or "").strip()
        final_refs = list(case.get("final_references") or [])
        prompt = (
            "Answer the USER QUESTION using ONLY the VERIFIED CONTEXT below. "
            "Do not introduce a Rule/Directive or factual claim absent from the context. "
            "When the context contains an applicable Rule/Directive reference, include that exact reference. "
            "For a yes/no compliance question, begin with Yes. or No. when the verified context determines the polarity. "
            "Use the same natural language as the USER QUESTION; do not switch to an unrelated language or script. "
            "If the context does not determine the answer, say Needs more context.\n\n"
            f"VERIFIED CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}"
        )
        started = time.perf_counter()
        raw_answer = _answer_text(client.generate(prompt)).strip()
        generated_seconds = time.perf_counter() - started

        final_answer, contract_changes = enforce_verified_answer_contract(
            answer=raw_answer,
            question=question,
            results=_synthetic_results(final_refs),
            yes_no_intent=bool(expected_prefix),
            require_reference=True,
        )
        refs = list(dict.fromkeys(_canonical_ref(m.group(0)) for m in REF_RE.finditer(final_answer)))
        wrong_refs = [ref for ref in refs if expected and ref != expected]
        ref_pass = expected in refs if expected else True
        polarity_pass = final_answer.casefold().startswith(expected_prefix.casefold()) if expected_prefix else True
        script_warning = bool(CJK_RE.search(final_answer)) and not bool(CJK_RE.search(question))
        generation_pass = ref_pass and polarity_pass and not wrong_refs and not script_warning
        results.append({
            "index": index,
            "question": question,
            "expected_reference": expected,
            "expected_prefix": expected_prefix,
            "final_references": final_refs,
            "seconds": round(generated_seconds, 4),
            "raw_answer": raw_answer,
            "final_answer": final_answer,
            "contract_changes": contract_changes,
            "reference_pass": ref_pass,
            "polarity_pass": polarity_pass,
            "wrong_references": wrong_refs,
            "language_script_warning_after_contract": script_warning,
            "generation_pass": generation_pass,
        })
        print(index, expected, f"{generated_seconds:.4f}s", "PASS" if generation_pass else "FAIL", final_answer)

    passed = sum(item["generation_pass"] for item in results)
    warnings = sum(item["language_script_warning_after_contract"] for item in results)
    overall = passed == len(results) and warnings == 0

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / f"answer_contract_live_{stamp}.json"
    report = {
        "version": "v6.5.5.7",
        "created": datetime.now().isoformat(timespec="seconds"),
        "overall": "PASS" if overall else "FAIL",
        "retrieval_evidence": str(retrieval_path),
        "model": OLLAMA_FAST_MODEL,
        "prewarm": prewarm,
        "cases": len(results),
        "passes": passed,
        "language_script_warnings_after_contract": warnings,
        "results": results,
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=" * 72)
    print(f"Overall: {report['overall']}")
    print(f"Answer contract: {passed}/{len(results)}")
    print(f"Language/script warnings after contract: {warnings}")
    print(out)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
