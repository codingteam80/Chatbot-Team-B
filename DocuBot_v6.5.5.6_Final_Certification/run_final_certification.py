from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

QA_DIR = Path(__file__).resolve().parent
ROOT = QA_DIR.parent
LOG_ROOT = ROOT / "logs"
PIPELINE_DIR = LOG_ROOT / "pipeline_diagnostics"
OUT_DIR = LOG_ROOT / "v6_5_5_final_certification"
LIFECYCLE_DIR = LOG_ROOT / "kb_lifecycle_probe"
HEALTH_DIR = LOG_ROOT / "kb_health"
TARGET_SECONDS = 25.0
QA_HARNESS_VERSION = "v6.5.5.6-final-certification"


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
    candidates: list[Path] = [summary_path]
    for key in ("benchmark_sheet", "retrieval_json", "llm_json", "lifecycle_json"):
        value = str(summary.get(key) or "").strip()
        if value:
            candidates.append(Path(value))
    health = _latest(HEALTH_DIR, "kb_health_*.txt")
    if health:
        candidates.append(health)
    validation = ROOT / "logs" / "v6_5_5_completion" / "validation_latest.json"
    if validation.is_file():
        candidates.append(validation)

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

    archive = OUT_DIR / f"DocuBot_v6.5.5.6_Final_Certification_Result_{stamp}.zip"
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


def _latency_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(item.get("combined_seconds") or 0.0) for item in cases]
    if not values:
        return {"count": 0, "passes": 0, "max_seconds": None, "cases": []}
    passes = sum(bool(item.get("within_target")) for item in cases)
    return {
        "count": len(cases),
        "passes": passes,
        "max_seconds": round(max(values), 4),
        "all_within_target": passes == len(cases),
        "cases": cases,
    }


def _finish(summary: dict[str, Any], stamp: str) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"final_certification_{stamp}.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    archive, sha_path = _bundle_evidence(path, summary, stamp)
    print("DOCUBOT FINAL CERTIFICATION v6.5.5.6")
    print("=" * 72)
    print("Overall:", summary.get("overall"))
    print("Message:", summary.get("message", ""))
    for goal, value in (summary.get("goals") or {}).items():
        print(f"{goal}: {value}")
    advisory = summary.get("ambiguous_multi_query_latency_advisory") or {}
    if advisory:
        print("ambiguous_multi_query_latency_advisory:", advisory.get("status"), f"({advisory.get('passes')}/{advisory.get('count')} <= {TARGET_SECONDS}s)")
    print("Report:", path)
    print("Evidence ZIP:", archive)
    print("SHA256:", sha_path)
    return 0 if summary.get("overall") == "PASS" else 1


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary: dict[str, Any] = {
        "version": "v6.5.5",
        "qa_harness_version": QA_HARNESS_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "overall": "FAIL",
        "target_normal_question_seconds": TARGET_SECONDS,
        "measurement_policy": {
            "normal_question": "single-query/early-accept path after production-equivalent startup prewarm",
            "ambiguous_multi_query": "reported separately as an optimization advisory; not part of the original normal-question latency gate",
            "startup_prewarm": "recorded separately and excluded from steady-state question timers",
        },
        "stages": {},
        "goals": {},
    }

    required = [
        ROOT / "venv" / "Scripts" / "python.exe",
        ROOT / "scripts" / "validate_v6_5_5_goal_completion.py",
        ROOT / "scripts" / "kb_health.py",
        ROOT / "scripts" / "test_incremental_kb_lifecycle_live.py",
        ROOT / "scripts" / "generate_v6_5_5_completion_benchmark.py",
        ROOT / "qa" / "v6_5_5_completion_benchmark_bank.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        summary["message"] = "Required current-project files are missing: " + "; ".join(missing)
        return _finish(summary, stamp)

    validator = _run([sys.executable, "-X", "utf8", "scripts/validate_v6_5_5_goal_completion.py"])
    summary["stages"]["static_validation"] = validator
    if validator["exit_code"] != 0:
        summary["message"] = "Static validation failed."
        return _finish(summary, stamp)

    health = _run([sys.executable, "-X", "utf8", "scripts/kb_health.py"])
    summary["stages"]["kb_health"] = health
    if health["exit_code"] != 0:
        summary["message"] = "KB Health failed."
        return _finish(summary, stamp)

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
        summary["message"] = "Live incremental lifecycle probe failed."
        return _finish(summary, stamp)

    benchmark_dir = ROOT / "logs" / "v6_5_5_completion"
    before_sheet = max((p.stat().st_mtime_ns for p in benchmark_dir.glob("benchmark_sheet_*_seed_6501.json")), default=0)
    generate = _run([
        sys.executable, "-X", "utf8", "scripts/generate_v6_5_5_completion_benchmark.py",
        "--seed", "6501", "--pairs", "6",
    ])
    summary["stages"]["benchmark_generation"] = generate
    sheet = _latest(benchmark_dir, "benchmark_sheet_*_seed_6501.json", after_ns=before_sheet)
    if generate["exit_code"] != 0 or sheet is None:
        summary["message"] = "Benchmark generation failed."
        return _finish(summary, stamp)
    summary["benchmark_sheet"] = str(sheet)

    before_retrieval = max((p.stat().st_mtime_ns for p in PIPELINE_DIR.glob("retrieval_steady_state_*.json")), default=0)
    retrieval = _run([
        sys.executable, "-X", "utf8", str(QA_DIR / "test_retrieval_steady_state.py"),
        "--benchmark-json", str(sheet),
    ])
    summary["stages"]["retrieval_steady_state"] = retrieval
    retrieval_json = _latest(PIPELINE_DIR, "retrieval_steady_state_*.json", after_ns=before_retrieval)
    retrieval_payload = _read_json(retrieval_json)
    summary["retrieval_json"] = str(retrieval_json) if retrieval_json else ""
    summary["retrieval_summary"] = {
        key: retrieval_payload.get(key)
        for key in (
            "overall", "prewarm", "case_count", "recall_passes", "precision_passes",
            "concept_consistency_passes", "concept_consistency_total",
            "multi_query_used_count", "multi_query_skipped_single_query_proven_count",
            "latency_seconds",
        )
    }
    summary["goals"]["retrieval_recall_precision_consistency"] = bool(
        retrieval["exit_code"] == 0 and retrieval_payload.get("overall") == "PASS"
    )
    if retrieval["exit_code"] != 0 or retrieval_json is None:
        summary["goals"]["answer_generation_from_verified_context"] = False
        summary["goals"]["normal_question_latency"] = False
        summary["message"] = "Retrieval gate failed; LLM generation was blocked."
        return _finish(summary, stamp)

    before_llm = max((p.stat().st_mtime_ns for p in PIPELINE_DIR.glob("llm_steady_state_*.json")), default=0)
    llm = _run([
        sys.executable, "-X", "utf8", str(QA_DIR / "test_llm_steady_state.py"),
        "--retrieval-json", str(retrieval_json),
    ])
    summary["stages"]["llm_steady_state"] = llm
    llm_json = _latest(PIPELINE_DIR, "llm_steady_state_*.json", after_ns=before_llm)
    llm_payload = _read_json(llm_json)
    summary["llm_json"] = str(llm_json) if llm_json else ""
    summary["llm_summary"] = {
        key: llm_payload.get(key)
        for key in (
            "overall", "prewarm", "case_count", "executed", "blocked",
            "generation_passes", "language_script_warnings", "latency_seconds",
        )
    }
    summary["goals"]["answer_generation_from_verified_context"] = bool(
        llm["exit_code"] == 0 and llm_payload.get("overall") == "PASS"
    )

    retrieval_cases = retrieval_payload.get("cases") or []
    llm_cases = llm_payload.get("cases") or []
    by_question = {str(item.get("question") or ""): item for item in llm_cases if isinstance(item, dict)}
    normal: list[dict[str, Any]] = []
    mq: list[dict[str, Any]] = []
    for item in retrieval_cases:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "")
        generation = by_question.get(question) or {}
        retrieval_seconds = float(item.get("seconds") or 0.0)
        generation_seconds = float(generation.get("seconds") or 0.0)
        total = retrieval_seconds + generation_seconds
        record = {
            "question": question,
            "retrieval_seconds": round(retrieval_seconds, 4),
            "generation_seconds": round(generation_seconds, 4),
            "combined_seconds": round(total, 4),
            "within_target": total <= TARGET_SECONDS,
            "multi_query_used": bool(item.get("multi_query_used")),
            "semantic_latency_decision": item.get("semantic_latency_decision", ""),
        }
        decision = str(record.get("semantic_latency_decision") or "")
        if record["multi_query_used"]:
            mq.append(record)
        elif decision == "multi_query_skipped_single_query_proven":
            normal.append(record)
        else:
            summary.setdefault("other_latency_cases", []).append(record)

    normal_summary = _latency_summary(normal)
    mq_summary = _latency_summary(mq)
    summary["normal_question_latency"] = {"target_seconds": TARGET_SECONDS, **normal_summary}
    summary["ambiguous_multi_query_latency_advisory"] = {
        "target_seconds": TARGET_SECONDS,
        "status": "PASS" if mq_summary.get("all_within_target") else "OPTIMIZATION_RECOMMENDED",
        **mq_summary,
    }
    summary["goals"]["normal_question_latency"] = bool(normal) and bool(normal_summary.get("all_within_target"))

    goal_values = list(summary["goals"].values())
    summary["overall"] = "PASS" if goal_values and all(goal_values) else "FAIL"
    if summary["overall"] == "PASS":
        if summary["ambiguous_multi_query_latency_advisory"]["status"] == "PASS":
            summary["message"] = "All original final quality goals passed; ambiguous MultiQuery latency also met the advisory target."
        else:
            summary["message"] = "All original final quality goals passed. Ambiguous MultiQuery latency remains a separate optimization advisory."
    else:
        summary["message"] = "One or more original final quality goals failed; review bundled evidence."
    return _finish(summary, stamp)


if __name__ == "__main__":
    raise SystemExit(main())
