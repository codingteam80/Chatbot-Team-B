from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "logs" / "mixed_goal_benchmark"
TEST_DIR = ROOT / "logs" / "test_evidence"
NO_RESULT = "Information not found in company knowledge base."
REF_PATTERN = re.compile(r"\b(?:Rule|Directive|Dir)\s+\d+(?:\.\d+)+\b", re.I)


def normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def canonical_ref(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    value = re.sub(r"^Dir\b", "Directive", value, flags=re.I)
    if value.lower().startswith("rule "):
        return "Rule " + value.split(" ", 1)[1]
    if value.lower().startswith("directive "):
        return "Directive " + value.split(" ", 1)[1]
    return value


def latest(pattern: str, folder: Path) -> Path | None:
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True) if folder.is_dir() else []
    return files[0] if files else None


def read_sources(raw: str) -> list[dict[str, Any]]:
    text = str(raw or "").strip()
    if not text or text.lower() == "nan":
        return []
    try:
        value = json.loads(text)
    except Exception:
        try:
            value = ast.literal_eval(text)
        except Exception:
            return []
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def split_run_blocks(log_text: str) -> dict[str, str]:
    blocks: dict[str, str] = {}
    pattern = re.compile(r"(?ms)^={70}\nTEST EXECUTION\n={70}\n(.*?)(?=^={70}\nTEST EXECUTION\n={70}\n|\Z)")
    for match in pattern.finditer(log_text):
        block = match.group(1)
        run_match = re.search(r"^Run ID\s*:\s*(\S+)", block, re.M)
        if run_match:
            blocks[run_match.group(1)] = block
    return blocks


def event_count(block: str, event_name: str) -> int:
    return len(re.findall(rf"^Event\s*:\s*{re.escape(event_name)}\s*$", block, re.M))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a completed v6.5.1 mixed manual benchmark from normal test evidence logs.")
    parser.add_argument("--sheet", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    sheet = args.sheet or latest("benchmark_sheet_*_seed_*.json", OUTDIR)
    csv_path = args.csv or latest("test_evidence_*.csv", TEST_DIR)
    log_path = args.log or latest("test_evidence_*.log", TEST_DIR)
    if not sheet or not sheet.is_file():
        print("[FAIL] Benchmark sheet JSON not found.")
        return 2
    if not csv_path or not csv_path.is_file() or not log_path or not log_path.is_file():
        print("[FAIL] test_evidence CSV/log not found.")
        return 3

    spec = json.loads(sheet.read_text(encoding="utf-8"))
    cases = list(spec.get("cases") or [])
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_question = {normalize(row.get("question")): row for row in rows}
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    blocks = split_run_blocks(log_text)

    results = []
    durations = []
    for case in cases:
        question = str(case.get("question", ""))
        expected = canonical_ref(case.get("expected_reference", ""))
        expected_prefix = str(case.get("expected_prefix", "") or "")
        row = by_question.get(normalize(question))
        if row is None:
            results.append({"question": question, "expected_reference": expected, "present": False})
            continue

        answer = str(row.get("actual_result", "") or "")
        fallback = normalize(row.get("fallback_used")) in {"true", "1", "yes"} or normalize(answer) == normalize(NO_RESULT)
        try:
            duration = float(row.get("duration_seconds") or 0.0)
        except Exception:
            duration = 0.0
        durations.append(duration)
        sources = read_sources(row.get("sources", ""))
        source_refs = sorted({canonical_ref(item.get("reference", "")) for item in sources if item.get("reference")})
        answer_refs = sorted({canonical_ref(item) for item in REF_PATTERN.findall(answer)})
        observed_refs = sorted(set(source_refs) | set(answer_refs))

        run_id = str(row.get("run_id", "") or "")
        block = blocks.get(run_id, "")
        pre_answer = block.split("ANSWER RESULT", 1)[0] if block else ""
        recall_pass = normalize(expected) in normalize(pre_answer)
        correctness_pass = (not fallback) and (expected in observed_refs or normalize(expected) in normalize(answer))
        if expected_prefix:
            correctness_pass = correctness_pass and answer.strip().casefold().startswith(expected_prefix.casefold())
        wrong_refs = [ref for ref in observed_refs if ref != expected]
        precision_pass = correctness_pass and not wrong_refs and (not source_refs or expected in source_refs)

        llm_calls = event_count(block, "LLM CALL LATENCY")
        mq_events = event_count(block, "MULTI-QUERY RETRIEVAL")
        semantic_events = event_count(block, "MISRA CORPUS SEMANTIC RULE RESOLUTION")
        duplicate_mq_guard = mq_events <= 2  # one generation plus at most one cached reuse event
        architecture_latency_pass = duration <= 25.0 and llm_calls <= 2 and duplicate_mq_guard

        results.append({
            "question": question,
            "concept_id": case.get("concept_id", ""),
            "expected_reference": expected,
            "expected_prefix": expected_prefix,
            "present": True,
            "answer": answer,
            "fallback": fallback,
            "source_references": source_refs,
            "answer_references": answer_refs,
            "wrong_references": wrong_refs,
            "recall_pass": recall_pass,
            "precision_pass": precision_pass,
            "correctness_pass": correctness_pass,
            "duration_seconds": duration,
            "llm_call_count": llm_calls,
            "multi_query_event_count": mq_events,
            "semantic_rule_event_count": semantic_events,
            "architecture_latency_pass": architecture_latency_pass,
        })

    complete = [item for item in results if item.get("present")]
    concepts: dict[str, list[dict[str, Any]]] = {}
    for item in complete:
        concepts.setdefault(str(item.get("concept_id", "")), []).append(item)
    consistency = {}
    for concept, items in concepts.items():
        if len(items) < 2:
            consistency[concept] = False
            continue
        refs = [tuple(item.get("source_references") or item.get("answer_references") or []) for item in items]
        consistency[concept] = all(item.get("correctness_pass") for item in items) and len(set(refs)) == 1

    total = len(cases)
    present_count = len(complete)
    correctness = sum(bool(item.get("correctness_pass")) for item in complete)
    recall = sum(bool(item.get("recall_pass")) for item in complete)
    precision = sum(bool(item.get("precision_pass")) for item in complete)
    latency_ok = sum(bool(item.get("architecture_latency_pass")) for item in complete)
    consistency_ok = sum(bool(value) for value in consistency.values())
    consistency_total = len(consistency)

    duration_stats = {
        "average": round(statistics.mean(durations), 4) if durations else None,
        "median": round(statistics.median(durations), 4) if durations else None,
        "max": round(max(durations), 4) if durations else None,
    }
    if durations:
        ordered = sorted(durations)
        idx = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
        duration_stats["p95"] = round(ordered[idx], 4)
    else:
        duration_stats["p95"] = None

    overall = (
        present_count == total
        and correctness == total
        and recall == total
        and precision == total
        and latency_ok == total
        and consistency_total > 0
        and consistency_ok == consistency_total
    )
    summary = {
        "version": "v6.5.1",
        "benchmark_seed": spec.get("seed"),
        "overall": "PASS" if overall else "FAIL",
        "questions_expected": total,
        "questions_present": present_count,
        "correctness_pass": correctness,
        "recall_pass": recall,
        "precision_pass": precision,
        "architecture_latency_pass": latency_ok,
        "consistency_pass": consistency_ok,
        "consistency_total": consistency_total,
        "latency_seconds": duration_stats,
        "criteria": {
            "correctness": "Expected Rule/Directive is used and no fallback occurs; polarity is checked where the source determines it.",
            "recall": "Expected Rule/Directive appears in pre-answer retrieval/semantic evidence.",
            "precision": "Final cited/mentioned structured reference is the expected one with no competing Rule/Directive reference.",
            "consistency": "Both natural variants for each selected concept converge to the same correct structured reference.",
            "latency": "No request exceeds 25s, no request uses more than two LLM calls, and MultiQuery is not repeatedly regenerated.",
        },
        "results": results,
        "source_csv": str(csv_path),
        "source_log": str(log_path),
        "sheet": str(sheet),
    }

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    json_out = OUTDIR / f"evaluation_{stamp}.json"
    txt_out = OUTDIR / f"evaluation_{stamp}.txt"
    json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "DOCUBOT v6.5.1 MIXED GOAL BENCHMARK EVALUATION",
        "=" * 72,
        f"Overall: {summary['overall']}",
        f"Questions: {present_count}/{total}",
        f"Correctness: {correctness}/{total}",
        f"Recall: {recall}/{total}",
        f"Precision: {precision}/{total}",
        f"Consistency: {consistency_ok}/{consistency_total}",
        f"Architecture latency: {latency_ok}/{total}",
        f"Latency avg/median/p95/max: {duration_stats['average']} / {duration_stats['median']} / {duration_stats['p95']} / {duration_stats['max']} sec",
        "",
    ]
    for i, item in enumerate(results, 1):
        if not item.get("present"):
            lines.append(f"{i}. MISSING | {item['expected_reference']} | {item['question']}")
            continue
        flags = [
            "C" if item.get("correctness_pass") else "c-",
            "R" if item.get("recall_pass") else "r-",
            "P" if item.get("precision_pass") else "p-",
            "L" if item.get("architecture_latency_pass") else "l-",
        ]
        lines.append(f"{i}. {'/'.join(flags)} | {item['duration_seconds']:.4f}s | {item['expected_reference']} | {item['question']}")
    txt_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"JSON: {json_out}")
    print(f"TXT: {txt_out}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
