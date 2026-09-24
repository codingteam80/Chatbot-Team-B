from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from config.settings import OLLAMA_FAST_MODEL
from llm.ollama_client import OllamaClient

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "logs" / "pipeline_diagnostics"


def main() -> int:
    parser = argparse.ArgumentParser(description="DocuBot LLM-only diagnostic over pre-verified retrieval context")
    parser.add_argument("--retrieval-json", required=True)
    parser.add_argument("--case", type=int, default=1, help="1-based case number from retrieval-only JSON")
    args = parser.parse_args()

    payload = json.loads(Path(args.retrieval_json).read_text(encoding="utf-8"))
    cases = payload.get("cases") or []
    index = max(1, int(args.case)) - 1
    if index >= len(cases):
        raise SystemExit("case index is outside retrieval JSON")
    case = cases[index]
    if case.get("retrieval_pass") is False:
        raise SystemExit("retrieval gate failed; LLM-only test intentionally blocked")

    question = str(case.get("question", "") or "").strip()
    context = str(case.get("verified_context", "") or "").strip()
    if not question or not context:
        raise SystemExit("retrieval JSON has no verified question/context")

    prompt = (
        "Answer the USER QUESTION using ONLY the VERIFIED CONTEXT below. "
        "Do not introduce a Rule/Directive or factual claim that is absent from the context. "
        "If the context does not determine the answer, say Needs more context.\n\n"
        f"VERIFIED CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}"
    )
    client = OllamaClient(model_name=OLLAMA_FAST_MODEL)
    started = time.perf_counter()
    answer = client.generate(prompt)
    seconds = time.perf_counter() - started

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / f"llm_from_verified_context_{stamp}.json"
    result = {
        "mode": "llm_only_from_verified_context",
        "question": question,
        "expected_reference": case.get("expected_reference", ""),
        "retrieved_references": case.get("final_references", []),
        "model": OLLAMA_FAST_MODEL,
        "seconds": round(seconds, 4),
        "answer": answer,
    }
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out)
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
