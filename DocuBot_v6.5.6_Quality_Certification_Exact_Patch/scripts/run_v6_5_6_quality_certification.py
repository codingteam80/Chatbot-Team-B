from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "logs" / "v6_5_6_quality_certification"
TARGET_SECONDS = 25.0


def _run(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        args,
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {"command": args, "exit_code": int(proc.returncode), "output": proc.stdout or ""}


def _latest(pattern: str, *, after_ns: int = 0) -> Path | None:
    files = [p for p in OUT_DIR.glob(pattern) if p.is_file() and p.stat().st_mtime_ns >= after_ns]
    return max(files, key=lambda p: p.stat().st_mtime_ns) if files else None


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _pctl(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return round(ordered[index], 4)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _finish(summary: dict[str, Any], stamp: str, evidence_files: list[Path]) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = OUT_DIR / f"v6.5.6_quality_certification_summary_{stamp}.json"
    summary["report_path"] = str(report)
    report.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    bundle = OUT_DIR / f"DocuBot_v6.5.6_Quality_Certification_Result_{stamp}.zip"
    unique_files: list[Path] = []
    seen = set()
    for path in [report, *evidence_files]:
        if path and path.is_file():
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_files.append(path)
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in unique_files:
            try:
                arcname = path.relative_to(ROOT).as_posix()
            except Exception:
                arcname = path.name
            archive.write(path, arcname=arcname)
    sha = _sha256(bundle)
    sha_path = bundle.with_suffix(bundle.suffix + ".sha256.txt")
    sha_path.write_text(f"{sha}  {bundle.name}\n", encoding="utf-8")

    print("DOCUBOT v6.5.6 QUALITY CERTIFICATION")
    print("=" * 72)
    print("Overall:", summary.get("overall"))
    print("Message:", summary.get("message", ""))
    for goal, value in (summary.get("goals") or {}).items():
        print(f"{goal}: {value}")
    print("Report:", report)
    print("Evidence ZIP:", bundle)
    print("SHA256:", sha_path)
    return 0 if summary.get("overall") in {"PASS", "PARTIAL_PASS"} else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run DocuBot v6.5.6 quality gates in one controlled sequence: static preservation -> KB Health -> "
            "actual add/modify/delete lifecycle -> semantic retrieval benchmark -> answer generation -> combined latency."
        )
    )
    parser.add_argument("--skip-live-delete", action="store_true", help="skip the actual temporary add/modify/delete lifecycle")
    parser.add_argument("--skip-answer", action="store_true", help="stop after retrieval benchmark")
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=6506)
    parser.add_argument("--full-bank", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evidence_files: list[Path] = []
    summary: dict[str, Any] = {
        "version": "v6.5.6",
        "base": "exact current working v6.5.5.4",
        "created": datetime.now().isoformat(timespec="seconds"),
        "overall": "FAIL",
        "target_normal_question_seconds": TARGET_SECONDS,
        "stages": {},
        "goals": {
            "retrieval_recall_precision_semantic_consistency": False,
            "answer_generation_after_verified_retrieval": False,
            "normal_question_latency": False,
            "delete_lifecycle_actual": False,
        },
        "preservation": {
            "production_runtime_intentionally_changed": False,
            "models_chunking_threshold_topk_intentionally_changed": False,
        },
    }

    validator = _run([sys.executable, "-X", "utf8", "-m", "scripts.validate_v6_5_6_quality_certification"])
    summary["stages"]["static_validation"] = validator
    if validator["exit_code"] != 0:
        summary["message"] = "Static preservation/quality-harness validation failed; no live KB mutation was attempted."
        return _finish(summary, stamp, evidence_files)

    health_before = _run([sys.executable, "-X", "utf8", "-m", "scripts.kb_health"])
    summary["stages"]["kb_health_before"] = health_before
    if health_before["exit_code"] != 0:
        summary["message"] = "KB Health failed before certification; no live KB mutation was attempted."
        return _finish(summary, stamp, evidence_files)

    if args.skip_live_delete:
        summary["goals"]["delete_lifecycle_actual"] = None
    else:
        before = max((p.stat().st_mtime_ns for p in OUT_DIR.glob("v6.5.6_delete_lifecycle_*.json")), default=0)
        lifecycle = _run([sys.executable, "-X", "utf8", "-m", "scripts.test_v6_5_6_delete_lifecycle_live"])
        lifecycle_json = _latest("v6.5.6_delete_lifecycle_*.json", after_ns=before)
        lifecycle_payload = _read_json(lifecycle_json)
        summary["stages"]["actual_delete_lifecycle"] = lifecycle
        summary["delete_lifecycle_json"] = str(lifecycle_json) if lifecycle_json else ""
        summary["delete_lifecycle_summary"] = {
            "overall": lifecycle_payload.get("overall"),
            "baseline_preserved": lifecycle_payload.get("baseline_preserved"),
            "kb_health_exit_code": lifecycle_payload.get("kb_health_exit_code"),
        }
        if lifecycle_json:
            evidence_files.append(lifecycle_json)
        passed = lifecycle["exit_code"] == 0 and lifecycle_payload.get("overall") == "PASS"
        summary["goals"]["delete_lifecycle_actual"] = bool(passed)
        if not passed:
            summary["message"] = "Actual delete lifecycle failed or could not prove clean restoration; retrieval/answer certification was stopped."
            return _finish(summary, stamp, evidence_files)

    before_sheet = max((p.stat().st_mtime_ns for p in OUT_DIR.glob("benchmark_sheet_*_seed_*.json")), default=0)
    generate_cmd = [
        sys.executable, "-X", "utf8", "-m", "scripts.generate_v6_5_6_quality_benchmark",
        "--seed", str(args.seed), "--pairs", str(args.pairs),
    ]
    if args.full_bank:
        generate_cmd.append("--full-bank")
    generation = _run(generate_cmd)
    summary["stages"]["benchmark_generation"] = generation
    sheet = _latest(f"benchmark_sheet_*_seed_{args.seed}.json", after_ns=before_sheet)
    if generation["exit_code"] != 0 or sheet is None:
        summary["message"] = "Quality benchmark generation failed."
        return _finish(summary, stamp, evidence_files)
    evidence_files.append(sheet)
    txt_sheet = sheet.with_suffix(".txt")
    if txt_sheet.is_file():
        evidence_files.append(txt_sheet)
    summary["benchmark_sheet"] = str(sheet)

    before_retrieval = max((p.stat().st_mtime_ns for p in OUT_DIR.glob("v6.5.6_retrieval_benchmark_*.json")), default=0)
    retrieval = _run([
        sys.executable, "-X", "utf8", "-m", "scripts.test_v6_5_6_retrieval_quality",
        "--benchmark-json", str(sheet),
    ])
    retrieval_json = _latest("v6.5.6_retrieval_benchmark_*.json", after_ns=before_retrieval)
    retrieval_payload = _read_json(retrieval_json)
    summary["stages"]["retrieval_quality"] = retrieval
    summary["retrieval_json"] = str(retrieval_json) if retrieval_json else ""
    summary["retrieval_summary"] = {
        key: retrieval_payload.get(key)
        for key in (
            "overall", "case_count", "recall_passes", "precision_passes", "top1_passes",
            "concept_consistency_passes", "concept_consistency_total", "multi_query_used_count",
            "multi_query_skipped_single_query_proven_count", "multi_query_exactly_two_alternatives_passes",
            "metrics", "latency_seconds",
        )
    }
    if retrieval_json:
        evidence_files.append(retrieval_json)
    retrieval_pass = retrieval["exit_code"] == 0 and retrieval_payload.get("overall") == "PASS"
    summary["goals"]["retrieval_recall_precision_semantic_consistency"] = bool(retrieval_pass)
    if not retrieval_pass:
        summary["message"] = "Retrieval recall/precision/semantic-consistency gate failed. Answer generation was intentionally blocked."
        return _finish(summary, stamp, evidence_files)

    if args.skip_answer:
        summary["goals"]["answer_generation_after_verified_retrieval"] = None
        summary["goals"]["normal_question_latency"] = None
        goal_values = [v for v in summary["goals"].values() if v is not None]
        summary["overall"] = "PARTIAL_PASS" if goal_values and all(goal_values) else "FAIL"
        summary["message"] = "Retrieval and requested live lifecycle gates passed; answer/combined-latency stage was skipped by request."
        return _finish(summary, stamp, evidence_files)

    before_answer = max((p.stat().st_mtime_ns for p in OUT_DIR.glob("v6.5.6_answer_diagnostic_*.json")), default=0)
    answer = _run([
        sys.executable, "-X", "utf8", "-m", "scripts.test_v6_5_6_answer_quality",
        "--retrieval-json", str(retrieval_json),
    ])
    answer_json = _latest("v6.5.6_answer_diagnostic_*.json", after_ns=before_answer)
    answer_payload = _read_json(answer_json)
    summary["stages"]["answer_quality"] = answer
    summary["answer_json"] = str(answer_json) if answer_json else ""
    summary["answer_summary"] = {
        key: answer_payload.get(key)
        for key in (
            "overall", "case_count", "executed", "blocked", "generation_passes",
            "claim_grounding_passes", "latency_seconds",
        )
    }
    if answer_json:
        evidence_files.append(answer_json)
    answer_pass = answer["exit_code"] == 0 and answer_payload.get("overall") == "PASS"
    summary["goals"]["answer_generation_after_verified_retrieval"] = bool(answer_pass)

    retrieval_cases = retrieval_payload.get("cases") or []
    answer_cases = answer_payload.get("cases") or []
    answer_by_question = {str(i.get("question") or ""): i for i in answer_cases if isinstance(i, dict)}
    combined: list[dict[str, Any]] = []
    for item in retrieval_cases:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "")
        generated = answer_by_question.get(question) or {}
        retrieval_seconds = float(item.get("seconds") or 0.0)
        answer_seconds = float(generated.get("seconds") or 0.0)
        total = retrieval_seconds + answer_seconds
        combined.append({
            "question": question,
            "concept_id": item.get("concept_id", ""),
            "variant": item.get("variant", ""),
            "retrieval_seconds": round(retrieval_seconds, 4),
            "answer_seconds": round(answer_seconds, 4),
            "combined_seconds": round(total, 4),
            "within_target": total <= TARGET_SECONDS,
            "multi_query_used": bool(item.get("multi_query_used")),
            "single_query_early_accept": bool(item.get("multi_query_skipped_single_query_proven")),
            "semantic_latency_decision": item.get("semantic_latency_decision", ""),
        })
    totals = [float(i["combined_seconds"]) for i in combined]
    passes = sum(bool(i["within_target"]) for i in combined)
    summary["combined_latency"] = {
        "target_seconds": TARGET_SECONDS,
        "passes": passes,
        "total": len(combined),
        "average": round(statistics.mean(totals), 4) if totals else None,
        "p50": round(statistics.median(totals), 4) if totals else None,
        "p95": _pctl(totals, 0.95),
        "max": round(max(totals), 4) if totals else None,
        "cases": combined,
    }
    summary["goals"]["normal_question_latency"] = bool(combined) and passes == len(combined)

    health_after = _run([sys.executable, "-X", "utf8", "-m", "scripts.kb_health"])
    summary["stages"]["kb_health_after"] = health_after
    post_health_pass = health_after["exit_code"] == 0

    goal_values = [value for value in summary["goals"].values() if value is not None]
    summary["overall"] = "PASS" if goal_values and all(goal_values) and post_health_pass else "FAIL"
    summary["message"] = (
        "Retrieval quality, verified-context answer generation, normal-question latency, actual delete lifecycle, and final KB Health all passed."
        if summary["overall"] == "PASS"
        else "One or more v6.5.6 quality goals remain failing; review the bundled per-stage evidence."
    )
    return _finish(summary, stamp, evidence_files)


if __name__ == "__main__":
    raise SystemExit(main())
