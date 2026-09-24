from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "qa" / "v6_5_5_completion_benchmark_bank.json"
OUTDIR = ROOT / "logs" / "v6_5_5_completion"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a reproducible v6.5.5 completion benchmark sheet.")
    parser.add_argument("--seed", type=int, default=6501)
    parser.add_argument("--pairs", type=int, default=6)
    args = parser.parse_args()

    bank = json.loads(BANK.read_text(encoding="utf-8"))
    rows = list(bank.get("cases") or [])
    by_concept: dict[str, list[dict]] = {}
    for row in rows:
        by_concept.setdefault(str(row.get("concept_id", "")), []).append(row)

    concepts = sorted(key for key, values in by_concept.items() if key and len(values) >= 2)
    pair_count = max(1, min(int(args.pairs), len(concepts)))
    rng = random.Random(int(args.seed))
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
        "version": "v6.5.5",
        "benchmark": "Goal Completion Benchmark",
        "seed": int(args.seed),
        "pair_count": pair_count,
        "question_count": len(selected),
        "production_runtime_modified": False,
        "questions_registered_in_production_qa": False,
        "cases": selected,
    }
    json_path = OUTDIR / f"benchmark_sheet_{stamp}_seed_{args.seed}.json"
    txt_path = OUTDIR / f"benchmark_sheet_{stamp}_seed_{args.seed}.txt"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "DOCUBOT v6.5.5 GOAL COMPLETION BENCHMARK",
        "=" * 72,
        f"Seed: {args.seed}",
        f"Concept pairs: {pair_count}",
        f"Questions: {len(selected)}",
        "",
        "QUESTIONS (expected references intentionally hidden from this TXT)",
    ]
    lines.extend(f"{index}. {row['question']}" for index, row in enumerate(selected, start=1))
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[PASS] v6.5.5 completion benchmark generated.")
    print(f"JSON: {json_path}")
    print(f"TXT: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
