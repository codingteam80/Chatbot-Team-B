from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "qa" / "v6_5_6_quality_benchmark_bank.json"
OUTDIR = ROOT / "logs" / "v6_5_6_quality_certification"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a reproducible DocuBot v6.5.6 quality benchmark sheet.")
    parser.add_argument("--seed", type=int, default=6506)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--full-bank", action="store_true", help="use every concept pair in the bank")
    args = parser.parse_args()

    bank = json.loads(BANK.read_text(encoding="utf-8"))
    rows = list(bank.get("cases") or [])
    by_concept: dict[str, list[dict]] = {}
    for row in rows:
        concept = str(row.get("concept_id", "")).strip()
        if concept:
            by_concept.setdefault(concept, []).append(row)

    concepts = sorted(key for key, values in by_concept.items() if len(values) >= 2)
    if not concepts:
        raise SystemExit("benchmark bank contains no concept pairs")

    pair_count = len(concepts) if args.full_bank else max(1, min(int(args.pairs), len(concepts)))
    rng = random.Random(int(args.seed))
    selected_concepts = list(concepts) if args.full_bank else rng.sample(concepts, pair_count)

    selected: list[dict] = []
    for concept in selected_concepts:
        variants = list(by_concept[concept])
        rng.shuffle(variants)
        selected.extend(variants[:2])
    rng.shuffle(selected)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v6.5.6",
        "benchmark": "Quality Certification Benchmark",
        "seed": int(args.seed),
        "pair_count": pair_count,
        "question_count": len(selected),
        "normal_question_target_seconds": float(bank.get("normal_question_target_seconds") or 25.0),
        "production_runtime_modified": False,
        "questions_registered_in_production_qa": False,
        "expected_reference_scope": "test-only",
        "cases": selected,
    }
    json_path = OUTDIR / f"benchmark_sheet_{stamp}_seed_{args.seed}.json"
    txt_path = OUTDIR / f"benchmark_sheet_{stamp}_seed_{args.seed}.txt"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "DOCUBOT v6.5.6 QUALITY CERTIFICATION BENCHMARK",
        "=" * 72,
        f"Seed: {args.seed}",
        f"Concept pairs: {pair_count}",
        f"Questions: {len(selected)}",
        f"Normal-question target: {payload['normal_question_target_seconds']} sec",
        "",
        "QUESTIONS (expected references intentionally hidden from this TXT)",
    ]
    lines.extend(f"{index}. {row['question']}" for index, row in enumerate(selected, start=1))
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[PASS] v6.5.6 quality benchmark generated.")
    print(f"JSON: {json_path}")
    print(f"TXT: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
