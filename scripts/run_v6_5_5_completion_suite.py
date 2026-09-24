from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "logs"
PIPELINE_DIR = LOG_ROOT / "pipeline_diagnostics"
BENCHMARK_DIR = LOG_ROOT / "v6_5_5_completion"
OUT_DIR = LOG_ROOT / "v6_5_5_completion"
LIFECYCLE_DIR = LOG_ROOT / "kb_lifecycle_probe"
HEALTH_DIR = LOG_ROOT / "kb_health"
TARGET_SECONDS = 25.0
QA_HARNESS_VERSION = "v6.5.5.5-final-quality-check"


def _latest(folder: Path, pattern: str, *, after_ns: int = 0) -> Path | None:
    files = [p for p in folder.glob(pattern) if p.is_file() and p.stat().st_mtime_ns >= after_ns]
    return max(files, key=lambda p: p.stat().st_mtime_ns) if files else None


def _run(args: list[str]) -> dict[str, Any]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing if existing else "")
    proc = subprocess.run(
        args,
        cwd=str(ROOT),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {"command": args, "exit_code": int(proc.returncode), "output": proc.stdout or ""}


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bundle_evidence(summary_path: Path, summary: dict[str, Any], stamp: str) -> tuple[Path, Path]:
    candidates: list[Path] = [summary_path, OUT_DIR / "validation_latest.json"]
    for key in ("benchmark_sheet", "retrieval_json", "llm_json"):
        value = str(summary.get(key) or "").strip()
        if value:
            candidates.append(Path(value))
    lifecycle = _latest(LIFECYCLE_DIR, "incremental_lifecycle_*.json")
    health = _latest(HEALTH_DIR, "kb_health_*.txt")
    if lifecycle:
        candidates.append(lifecycle)
    if health:
        candidates.append(health)

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        key = str(resolved).casefold()
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        unique.append(path)

    archive = OUT_DIR / f"DocuBot_v6.5.5.5_Final_Quality_Result_{stamp}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in unique:
            try:
                arcname = str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
            except Exception:
                arcname = path.name
            zf.write(path, arcname=arcname)

    sha_path = archive.with_suffix(archive.suffix + ".sha256.txt")
    sha_path.write_text(f"{_sha256(archive)}  {archive.name}\n", encoding="utf-8")
    return archive, sha_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final DocuBot quality gates in order: static validation -> KB Health -> "
            "live add/modify/delete lifecycle -> retrieval recall/precision/semantic consistency -> "
            "LLM generation from verified context -> combined normal-question latency."
        )
    )
    parser.add_argument(
        "--with-live-kb-lifecycle",
        action="store_true",
        help="run a temporary transactional add/modify/delete probe and restore the clean KB state",
    )
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary: dict[str, Any] = {
        "version": "v6.5.5",
        "qa_harness_version": QA_HARNESS_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "overall": "FAIL",
        "target_normal_question_seconds": TARGET_SECONDS,
        "stages": {},
        "goals": {},
    }

    validator = _run([sys.executable, "-X", "utf8", "scripts/validate_v6_5_5_goal_completion.py"])
    summary["stages"]["static_validation"] = validator
    if validator["exit_code"] != 0:
        summary["message"] = "Static validation failed; runtime diagnostics were not started."
        return _finish(summary, stamp)

    health = _run([sys.executable, "-X", "utf8", "scripts/kb_health.py"])
    summary["stages"]["kb_health"] = health
    if health["exit_code"] != 0:
        summary["message"] = "KB Health failed; retrieval/LLM diagnostics were not started."
        return _finish(summary, stamp)

    if args.with_live_kb_lifecycle:
        before_lifecycle = max((p.stat().st_mtime_ns for p in LIFECYCLE_DIR.glob("incremental_lifecycle_*.json")), default=0)
        lifecycle = _run([sys.executable, "-X", "utf8", "scripts/test_incremental_kb_lifecycle_live.py"])
        summary["stages"]["live_incremental_lifecycle"] = lifecycle
        lifecycle_json = _latest(LIFECYCLE_DIR, "incremental_lifecycle_*.json", after_ns=before_lifecycle)
        lifecycle_payload = _read_json(lifecycle_json)
        summary["lifecycle_json"] = str(lifecycle_json) if lifecycle_json else ""
        summary["lifecycle_summary"] = {
            "overall": lifecycle_payload.get("overall"),
            "message": lifecycle_payload.get("message"),
            "preservation": lifecycle_payload.get("preservation"),
        }
        summary["goals"]["delete_lifecycle_actual"] = bool(
            lifecycle["exit_code"] == 0
            and lifecycle_payload.get("overall") == "PASS"
            and ((lifecycle_payload.get("preservation") or {}).get("all_pass") is True)
        )
        if lifecycle["exit_code"] != 0:
            summary["message"] = "Live incremental lifecycle probe failed; retrieval diagnostics were not started."
            return _finish(summary, stamp)
    else:
        summary["goals"]["delete_lifecycle_actual"] = None

    before_sheet = max((p.stat().st_mtime_ns for p in BENCHMARK_DIR.glob("benchmark_sheet_*_seed_6501.json")), default=0)
    generate = _run([
        sys.executable,
        "-X",
        "utf8",
        "scripts/generate_v6_5_5_completion_benchmark.py",
        "--seed",
        "6501",
        "--pairs",
        "6",
    ])
    summary["stages"]["benchmark_generation"] = generate
    sheet = _latest(BENCHMARK_DIR, "benchmark_sheet_*_seed_6501.json", after_ns=before_sheet)
    if generate["exit_code"] != 0 or sheet is None:
        summary["message"] = "Seed-6501 benchmark generation failed."
        return _finish(summary, stamp)
    summary["benchmark_sheet"] = str(sheet)

    before_retrieval = max((p.stat().st_mtime_ns for p in PIPELINE_DIR.glob("retrieval_only_*.json")), default=0)
    retrieval = _run([
        sys.executable,
        "-X",
        "utf8",
        "scripts/test_retrieval_pipeline.py",
        "--benchmark-json",
        str(sheet),
    ])
    summary["stages"]["retrieval_only"] = retrieval
    retrieval_json = _latest(PIPELINE_DIR, "retrieval_only_*.json", after_ns=before_retrieval)
    retrieval_payload = _read_json(retrieval_json)
    summary["retrieval_json"] = str(retrieval_json) if retrieval_json else ""
    summary["retrieval_summary"] = {
        key: retrieval_payload.get(key)
        for key in (
            "overall",
            "case_count",
            "recall_passes",
            "precision_passes",
            "concept_consistency_passes",
            "concept_consistency_total",
            "multi_query_used_count",
            "multi_query_skipped_single_query_proven_count",
            "latency_seconds",
        )
    }
    case_count = int(retrieval_payload.get("case_count") or 0)
    summary["goals"]["retrieval_recall_precision_consistency"] = bool(
        retrieval["exit_code"] == 0
        and retrieval_payload.get("overall") == "PASS"
        and case_count > 0
    )

    if retrieval["exit_code"] != 0 or retrieval_json is None:
        summary["message"] = "Retrieval-only gate failed. LLM generation was intentionally blocked."
        summary["goals"]["answer_generation_from_verified_context"] = False
        summary["goals"]["normal_question_latency"] = False
        return _finish(summary, stamp)

    if args.skip_llm:
        summary["goals"]["answer_generation_from_verified_context"] = None
        summary["goals"]["normal_question_latency"] = None
        summary["overall"] = "PARTIAL_PASS"
        summary["message"] = "Retrieval gate passed; LLM stage skipped by request."
        return _finish(summary, stamp)

    before_llm = max((p.stat().st_mtime_ns for p in PIPELINE_DIR.glob("llm_from_verified_context_*.json")), default=0)
    llm = _run([
        sys.executable,
        "-X",
        "utf8",
        "scripts/test_llm_from_verified_context.py",
        "--retrieval-json",
        str(retrieval_json),
    ])
    summary["stages"]["llm_from_verified_context"] = llm
    llm_json = _latest(PIPELINE_DIR, "llm_from_verified_context_*.json", after_ns=before_llm)
    llm_payload = _read_json(llm_json)
    summary["llm_json"] = str(llm_json) if llm_json else ""
    summary["llm_summary"] = {
        key: llm_payload.get(key)
        for key in (
            "overall",
            "case_count",
            "executed",
            "blocked",
            "generation_passes",
            "latency_seconds",
        )
    }
    summary["goals"]["answer_generation_from_verified_context"] = bool(
        llm["exit_code"] == 0 and llm_payload.get("overall") == "PASS"
    )

    retrieval_cases = retrieval_payload.get("cases") or []
    llm_cases = llm_payload.get("cases") or []
    by_question = {str(item.get("question") or ""): item for item in llm_cases if isinstance(item, dict)}
    combined = []
    for item in retrieval_cases:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "")
        generation = by_question.get(question) or {}
        retrieval_seconds = float(item.get("seconds") or 0.0)
        generation_seconds = float(generation.get("seconds") or 0.0)
        total = retrieval_seconds + generation_seconds
        combined.append({
            "question": question,
            "retrieval_seconds": round(retrieval_seconds, 4),
            "generation_seconds": round(generation_seconds, 4),
            "combined_seconds": round(total, 4),
            "within_target": total <= TARGET_SECONDS,
            "multi_query_used": bool(item.get("multi_query_used")),
            "semantic_latency_decision": item.get("semantic_latency_decision", ""),
        })
    latency_passes = sum(item["within_target"] for item in combined)
    summary["combined_latency"] = {
        "target_seconds": TARGET_SECONDS,
        "passes": latency_passes,
        "total": len(combined),
        "max_seconds": round(max((item["combined_seconds"] for item in combined), default=0.0), 4),
        "cases": combined,
    }
    summary["goals"]["normal_question_latency"] = bool(combined) and latency_passes == len(combined)

    goal_values = [value for value in summary["goals"].values() if value is not None]
    summary["overall"] = "PASS" if goal_values and all(goal_values) else "FAIL"
    summary["message"] = (
        "All executed final quality gates passed."
        if summary["overall"] == "PASS"
        else "One or more final quality goals still need attention; review the bundled evidence."
    )
    return _finish(summary, stamp)


def _finish(summary: dict[str, Any], stamp: str) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"completion_{stamp}.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    archive, sha_path = _bundle_evidence(path, summary, stamp)
    print("DOCUBOT FINAL QUALITY CHECK")
    print("=" * 72)
    print("Overall:", summary.get("overall"))
    print("Message:", summary.get("message", ""))
    for goal, value in (summary.get("goals") or {}).items():
        print(f"{goal}: {value}")
    print("Report:", path)
    print("Evidence ZIP:", archive)
    print("SHA256:", sha_path)
    return 0 if summary.get("overall") in {"PASS", "PARTIAL_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
