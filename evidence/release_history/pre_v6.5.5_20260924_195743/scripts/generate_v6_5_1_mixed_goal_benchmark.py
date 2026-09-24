from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "qa" / "v6_5_1_mixed_goal_benchmark_bank.json"
OUTDIR = ROOT / "logs" / "mixed_goal_benchmark"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a reproducible mixed unseen manual benchmark sheet.")
    parser.add_argument("--seed", type=int, default=None, help="Optional reproducible random seed.")
    parser.add_argument("--pairs", type=int, default=None, help="Number of concept pairs to select.")
    args = parser.parse_args()

    bank = json.loads(BANK.read_text(encoding="utf-8"))
    rows = list(bank.get("cases") or [])
    by_concept: dict[str, list[dict]] = {}
    for row in rows:
        by_concept.setdefault(str(row.get("concept_id", "")), []).append(row)

    concepts = sorted(key for key, values in by_concept.items() if key and len(values) >= 2)
    pair_count = int(args.pairs or bank.get("default_pairs") or 6)
    pair_count = max(1, min(pair_count, len(concepts)))
    seed = int(args.seed if args.seed is not None else datetime.now().strftime("%Y%m%d%H%M%S"))
    rng = random.Random(seed)
    selected_concepts = rng.sample(concepts, pair_count)

    selected: list[dict] = []
    for concept in selected_concepts:
        variants = list(by_concept[concept])
        rng.shuffle(variants)
        selected.extend(variants[:2])
    rng.shuffle(selected)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v6.5.1",
        "benchmark": "Mixed Goal Benchmark",
        "seed": seed,
        "pair_count": pair_count,
        "question_count": len(selected),
        "production_runtime_modified": False,
        "questions_registered_in_production_qa": False,
        "cases": selected,
    }
    json_path = OUTDIR / f"benchmark_sheet_{stamp}_seed_{seed}.json"
    txt_path = OUTDIR / f"benchmark_sheet_{stamp}_seed_{seed}.txt"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "DOCUBOT v6.5.1 MIXED GOAL BENCHMARK",
        "=" * 72,
        f"Seed: {seed}",
        f"Concept pairs: {pair_count}",
        f"Questions: {len(selected)}",
        "",
        "MANUAL EXECUTION RULES",
        "- Use a fresh chat for every question.",
        "- Ask each question exactly as written.",
        "- Do not reveal the expected Rule/Directive to DocuBot.",
        "- Keep TEST_EVIDENCE_MODE enabled.",
        "- After all questions, keep the generated test_evidence .log and .csv.",
        "",
        "QUESTIONS",
    ]
    for index, row in enumerate(selected, start=1):
        lines.append(f"{index}. {row['question']}")
    lines.extend([
        "",
        "Blind-test note: expected references are intentionally omitted from this TXT.",
        "The evaluator reads them only from the companion JSON after execution.",
    ])
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[PASS] Mixed benchmark sheet generated.")
    print(f"Seed: {seed}")
    print(f"Questions: {len(selected)}")
    print(f"TXT: {txt_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
