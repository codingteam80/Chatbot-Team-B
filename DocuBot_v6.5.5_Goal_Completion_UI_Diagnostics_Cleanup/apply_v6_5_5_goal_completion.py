from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

PATCH = Path(__file__).resolve().parent
PAYLOAD = PATCH / "_payload"
PAYLOAD_HASHES = json.loads((PATCH / "payload_hashes.json").read_text(encoding="utf-8"))
BASELINE = json.loads((PATCH / "baseline_hashes.json").read_text(encoding="utf-8"))
SIMULATE = os.getenv("DOCUBOT_V655_SIMULATE", "").strip() == "1"
FORCE_VALIDATION_FAIL = os.getenv("DOCUBOT_V655_FORCE_VALIDATION_FAIL", "").strip() == "1"
VERSION = "v6.5.5"
RESULT_DIR_REL = "logs/v6_5_5_completion"
MARKER_REL = "GOAL_COMPLETION_APPLIED_v6.5.5.txt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_project(arg: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if arg:
        candidates.append(Path(arg).expanduser())
    candidates += [
        PATCH.parent,
        PATCH.parent / "company-chatbot",
        PATCH.parent.parent,
        PATCH.parent.parent / "company-chatbot",
        Path.cwd(),
        Path.cwd() / "company-chatbot",
    ]
    seen: set[str] = set()
    for item in candidates:
        try:
            root = item.resolve()
        except Exception:
            continue
        key = str(root).casefold()
        if key in seen:
            continue
        seen.add(key)
        if (root / "run.py").is_file() and (root / "config" / "settings.py").is_file():
            return root
    return None


def python_candidates(root: Path):
    for path in (
        root / "venv" / "Scripts" / "python.exe",
        root / ".venv" / "Scripts" / "python.exe",
        root / "env" / "Scripts" / "python.exe",
    ):
        if path.is_file():
            yield str(path)
    if SIMULATE:
        yield sys.executable


def run_cmd(root: Path, args: list[str], env: dict[str, str] | None = None):
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        args,
        cwd=str(root),
        env=merged,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def run_module(root: Path, python_exe: str, module: str, *args: str):
    return run_cmd(root, [python_exe, "-X", "utf8", "-u", "-m", module, *args])


def choose_python(root: Path):
    attempts: list[dict[str, Any]] = []
    for python_exe in python_candidates(root):
        if SIMULATE and python_exe == sys.executable:
            return python_exe, "SIMULATION: dependency probe bypassed.", attempts
        probe = run_cmd(
            root,
            [
                python_exe,
                "-X",
                "utf8",
                "-u",
                "-c",
                (
                    "import sys,qdrant_client,streamlit,rank_bm25,numpy; "
                    "print(sys.executable); print('qdrant_client=OK'); "
                    "print('streamlit=OK'); print('rank_bm25=OK'); print('numpy=OK')"
                ),
            ],
        )
        attempts.append(
            {"python": python_exe, "returncode": probe.returncode, "output": (probe.stdout or "").strip()}
        )
        if probe.returncode == 0:
            return python_exe, (probe.stdout or "").strip(), attempts
    return None, "", attempts


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return True


def inspect_existing_kb_lock(root: Path):
    """Fail closed if a real update is active; only remove a provably stale lock."""

    lock_path = root / "logs" / "kb_update" / "kb_update.lock"
    if not lock_path.is_file():
        return True, "No active KB update lock."
    try:
        raw = json.loads(lock_path.read_text(encoding="utf-8"))
        existing = raw if isinstance(raw, dict) else {}
    except Exception:
        existing = {}
    try:
        pid = int(existing.get("pid", 0) or 0)
    except Exception:
        pid = 0
    if pid_is_running(pid):
        return False, (
            "A live DocuBot KB-update process owns logs\\kb_update\\kb_update.lock "
            f"(PID {pid}, source={existing.get('source', 'unknown')}, "
            f"role={existing.get('owner_role', 'unknown')}). Stop the update/DocuBot and rerun."
        )
    try:
        lock_path.unlink(missing_ok=True)
        return True, f"Removed stale KB update lock from dead PID {pid or 'unknown'}."
    except Exception as error:
        return False, f"Could not remove stale KB update lock: {type(error).__name__}: {error}"


def verify_payload():
    problems: list[str] = []
    for rel, wanted in PAYLOAD_HASHES.items():
        path = PAYLOAD / rel
        if not path.is_file():
            problems.append(f"missing payload: {rel}")
            continue
        actual = sha256_file(path).lower()
        if actual != str(wanted).lower():
            problems.append(f"payload hash mismatch: {rel} => {actual}")
    # Prevent unmanifested payload files from silently entering the project.
    actual_files = {
        p.relative_to(PAYLOAD).as_posix()
        for p in PAYLOAD.rglob("*")
        if p.is_file()
    }
    expected_files = set(PAYLOAD_HASHES)
    for rel in sorted(actual_files - expected_files):
        problems.append(f"unmanifested payload file: {rel}")
    for rel in sorted(expected_files - actual_files):
        problems.append(f"manifested payload file absent: {rel}")
    return not problems, problems


def verify_baseline(root: Path):
    problems: list[str] = []
    matched: dict[str, str] = {}
    for rel, wanted in (BASELINE.get("required_existing_sha256") or {}).items():
        path = root / rel
        if not path.is_file():
            problems.append(f"missing v6.5.4 baseline file: {rel}")
            continue
        actual = sha256_file(path).lower()
        if actual != str(wanted).lower():
            problems.append(f"baseline mismatch: {rel} => {actual}")
        else:
            matched[rel] = actual
    for rel in BASELINE.get("required_absent") or []:
        if (root / rel).exists():
            problems.append(f"expected absent on clean v6.5.4 baseline: {rel}")
    manifest = BASELINE.get("required_manifest") or {}
    manifest_rel = str(manifest.get("path") or "")
    manifest_sha = str(manifest.get("sha256") or "").lower()
    manifest_path = root / manifest_rel
    if not manifest_rel or not manifest_path.is_file():
        problems.append(f"missing exact v6.5.4 component manifest: {manifest_rel or '[unspecified]'}")
    elif sha256_file(manifest_path).lower() != manifest_sha:
        problems.append(f"v6.5.4 component manifest hash mismatch: {manifest_rel}")
    return not problems, problems, matched


def print_output(result) -> None:
    output = result.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n")


def read_cleanup_plan_from_payload() -> dict[str, Any]:
    path = PAYLOAD / "cleanup_plan_v6.5.5.json"
    return json.loads(path.read_text(encoding="utf-8"))


def all_tracked_paths(cleanup_plan: dict[str, Any]) -> list[str]:
    tracked = list(PAYLOAD_HASHES)
    tracked.append(str((BASELINE.get("required_manifest") or {}).get("path") or ""))
    tracked.append(MARKER_REL)
    for item in cleanup_plan.get("items") or []:
        rel = str(item.get("path") or "")
        if rel:
            tracked.append(rel)
    return [rel for rel in dict.fromkeys(tracked) if rel]


def backup_paths(root: Path, history: Path, rels: list[str]) -> dict[str, bool]:
    state: dict[str, bool] = {}
    for rel in rels:
        src = root / rel
        existed = src.is_file()
        state[rel] = existed
        if existed:
            backup = history / rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, backup)
    return state


def restore_paths(root: Path, history: Path, state: dict[str, bool]) -> None:
    for rel, existed in state.items():
        dst = root / rel
        backup = history / rel
        if existed and backup.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, dst)
        elif not existed:
            try:
                if dst.is_file() or dst.is_symlink():
                    dst.unlink(missing_ok=True)
            except Exception:
                pass


def cleanup_exact_legacy(root: Path, plan: dict[str, Any], stamp: str):
    archive_root_template = str(plan.get("archive_root") or "evidence/release_history/v6.5.5_final_cleanup_<timestamp>")
    archive_rel = archive_root_template.replace("<timestamp>", stamp)
    archive_root = root / archive_rel
    archived: list[dict[str, str]] = []
    skipped_modified: list[dict[str, str]] = []
    absent: list[str] = []

    for item in plan.get("items") or []:
        rel = str(item.get("path") or "")
        if not rel:
            continue
        src = root / rel
        if not src.is_file():
            absent.append(rel)
            continue
        allowed = []
        if item.get("sha256"):
            allowed.append(str(item["sha256"]).lower())
        allowed.extend(str(x).lower() for x in (item.get("allowed_sha256") or []))
        actual = sha256_file(src).lower()
        if allowed and actual not in allowed:
            skipped_modified.append({"path": rel, "sha256": actual})
            continue
        dst = archive_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        archived.append({"path": rel, "sha256": actual})

    report = {
        "version": VERSION,
        "archive_root": archive_rel.replace("\\", "/"),
        "archived_count": len(archived),
        "skipped_modified_count": len(skipped_modified),
        "absent_count": len(absent),
        "archived": archived,
        "skipped_modified": skipped_modified,
        "absent": absent,
    }
    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / "cleanup_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return archive_root, report


def completed(output: str = ""):
    return subprocess.CompletedProcess([], 0, stdout=output)


def fail_and_restore(
    root: Path,
    history: Path,
    state: dict[str, bool],
    cleanup_archive: Path | None,
    message: str,
    code: int,
) -> int:
    print("[FAIL]", message)
    print("[ROLLBACK] Restoring pre-v6.5.5 files...")
    restore_paths(root, history, state)
    if cleanup_archive and cleanup_archive.exists():
        shutil.rmtree(cleanup_archive, ignore_errors=True)
    print("[ROLLBACK] Completed. Previous project files were restored.")
    return code


def main() -> int:
    root = locate_project(sys.argv[1] if len(sys.argv) > 1 else None)
    if root is None:
        print("[FAIL] DocuBot project not found. No files changed.")
        return 2

    print("=" * 116)
    print("DocuBot v6.5.5 - Goal Completion: Stable KB UI + Diagnostics + Safe Final Cleanup")
    print("=" * 116)
    print("[INFO] Base required: exact verified v6.5.4.")
    print("[INFO] Preserves working incremental KB lifecycle: unchanged=SKIP; new/modified/deleted=PROCESS.")
    print("[INFO] Keeps maintenance area + welcome/chat composer in their normal positions during KB updates.")
    print("[INFO] Restores explicit confirmation checkbox for BOTH incremental and full-rebuild maintenance.")
    print("[INFO] Adds retrieval-only, LLM-from-verified-context, latency, and live add/modify/delete completion gates.")
    print("[INFO] Archives only exact known-obsolete files; modified/historical evidence is preserved.")
    print("[INFO] Installer itself does NOT run a production KB rebuild or the live benchmark suite.")

    safe, lock_note = inspect_existing_kb_lock(root)
    if not safe:
        print("[BLOCKED]", lock_note)
        print("[INFO] No project files were changed.")
        return 13
    print("[PASS]", lock_note)

    payload_ok, payload_problems = verify_payload()
    if not payload_ok:
        print("[FAIL] Payload integrity failed. No project files were changed.")
        for item in payload_problems:
            print(" -", item)
        return 4
    print(f"[PASS] Payload integrity verified ({len(PAYLOAD_HASHES)} files).")

    baseline_ok, baseline_problems, matched = verify_baseline(root)
    if not baseline_ok:
        print("[FAIL] Project is not the exact supported v6.5.4 baseline. No project files were changed.")
        for item in baseline_problems:
            print(" -", item)
        return 3
    print(f"[PASS] Exact v6.5.4 baseline verified ({len(matched)} covered files + component manifest).")

    python_exe, python_probe, attempts = choose_python(root)
    if not python_exe:
        print("[FAIL] No project Python with required DocuBot dependencies was found. No project files were changed.")
        for attempt in attempts:
            print(f" - {attempt['python']} => rc={attempt['returncode']}")
            if attempt.get("output"):
                print(attempt["output"])
        return 5
    print("[PASS] Project Python selected:", python_exe)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cleanup_plan = read_cleanup_plan_from_payload()
    history = root / "evidence" / "release_history" / f"pre_v6.5.5_{stamp}"
    history.mkdir(parents=True, exist_ok=True)
    tracked = all_tracked_paths(cleanup_plan)
    state = backup_paths(root, history, tracked)
    cleanup_archive: Path | None = None

    try:
        for src in PAYLOAD.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(PAYLOAD)
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    except Exception as error:
        return fail_and_restore(
            root, history, state, cleanup_archive,
            f"Payload copy failed: {type(error).__name__}: {error}", 6,
        )

    marker = root / MARKER_REL
    marker.write_text(
        "\n".join([
            "DocuBot v6.5.5 Goal Completion - UI + Diagnostics + Safe Cleanup",
            "=" * 100,
            f"Applied: {datetime.now().isoformat(timespec='seconds')}",
            "Base: exact verified v6.5.4",
            "KB incremental lifecycle preserved: YES",
            "UNCHANGED -> skip existing vectors: YES",
            "NEW/MODIFIED/DELETED -> process delta only: YES",
            "KB maintenance + chat/welcome layout stability fix: YES",
            "Confirmation checkbox for incremental + full rebuild: YES",
            "Retrieval-only recall/precision/semantic-consistency diagnostic: YES",
            "LLM-from-verified-context grounding/polarity diagnostic: YES",
            "Combined normal-question latency completion gate: 25 seconds",
            "Safe live add/modify/delete lifecycle probe: YES (manual, not installer)",
            "Known obsolete files cleanup policy: archive exact hashes only",
            "Historical test evidence preserved: YES",
            "Chunk size/overlap changed: NO (900/150)",
            "Global retrieval threshold changed: NO (0.55)",
            "Vector/BM25/Final Top-K changed: NO (10/10/3)",
            "MultiQuery alternative count changed: NO (2)",
            "Models changed: NO",
            "Production Qdrant/BM25 rebuilt by installer: NO",
            f"Pre-install backup: {history}",
        ]) + "\n",
        encoding="utf-8",
    )

    validation = run_cmd(
        root,
        [python_exe, "-X", "utf8", "-u", "scripts/validate_v6_5_5_goal_completion.py"],
    )
    print_output(validation)
    if FORCE_VALIDATION_FAIL:
        validation = subprocess.CompletedProcess(validation.args, 99, stdout=(validation.stdout or "") + "\nFORCED TEST FAILURE\n")
    if validation.returncode != 0:
        return fail_and_restore(root, history, state, cleanup_archive, "v6.5.5 static validation failed.", 7)

    if SIMULATE:
        semantic_source_check = completed("SIMULATION: semantic source/profile check skipped.\n")
        semantic_ready = completed("SIMULATION: semantic Rule index readiness check skipped.\n")
        kb_health = completed("SIMULATION: KB Health skipped.\n")
        portability = completed("SIMULATION: KB-update portability dry-run skipped.\n")
    else:
        semantic_source_check = run_module(root, python_exe, "scripts.build_misra_semantic_rule_index", "--check")
        print_output(semantic_source_check)
        if semantic_source_check.returncode != 0:
            return fail_and_restore(
                root, history, state, cleanup_archive,
                "MISRA semantic Rule source/profile check failed; no KB rebuild was attempted.", 8,
            )

        semantic_ready = run_cmd(
            root,
            [python_exe, "-X", "utf8", "-u", "-c", (
                "from retrieval.semantic_rule_resolver import semantic_rule_index_ready; "
                "import sys; ok=semantic_rule_index_ready(); "
                "print('semantic_rule_index_ready='+str(ok)); sys.exit(0 if ok else 3)"
            )],
        )
        print_output(semantic_ready)
        if semantic_ready.returncode != 0:
            return fail_and_restore(
                root, history, state, cleanup_archive,
                "Existing semantic Rule index is not ready; installer did not rebuild it.", 9,
            )

        kb_health = run_module(root, python_exe, "scripts.kb_health")
        print_output(kb_health)
        if kb_health.returncode != 0:
            return fail_and_restore(root, history, state, cleanup_archive, "Read-only KB Health failed.", 10)

        portability = run_module(root, python_exe, "scripts.test_kb_update_portability")
        print_output(portability)
        if portability.returncode != 0:
            return fail_and_restore(
                root, history, state, cleanup_archive,
                "KB update portability/read-only dry-run failed.", 11,
            )

    try:
        cleanup_archive, cleanup_report = cleanup_exact_legacy(root, cleanup_plan, stamp)
    except Exception as error:
        return fail_and_restore(
            root, history, state, cleanup_archive,
            f"Final cleanup archive step failed: {type(error).__name__}: {error}", 12,
        )

    # Final payload hashes must remain exact after cleanup.
    post_payload_problems: list[str] = []
    for rel, wanted in PAYLOAD_HASHES.items():
        path = root / rel
        if not path.is_file():
            post_payload_problems.append(f"missing installed payload: {rel}")
        elif sha256_file(path).lower() != str(wanted).lower():
            post_payload_problems.append(f"installed payload changed unexpectedly: {rel}")
    if post_payload_problems:
        for item in post_payload_problems:
            print(" -", item)
        return fail_and_restore(root, history, state, cleanup_archive, "Post-install payload integrity failed.", 14)

    outdir = root / RESULT_DIR_REL
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": VERSION,
        "overall": "PASS",
        "base": "exact verified v6.5.4",
        "selected_project_python": python_exe,
        "validation_exit_code": validation.returncode,
        "semantic_source_check_exit_code": semantic_source_check.returncode,
        "semantic_rule_index_ready": semantic_ready.returncode == 0,
        "kb_health_exit_code": kb_health.returncode,
        "kb_update_portability_dry_run_exit_code": portability.returncode,
        "ui_kb_update_layout_stability_fix": True,
        "incremental_and_full_rebuild_confirmation_checkbox": True,
        "kb_incremental_lifecycle_preserved": True,
        "unchanged_files_reembedded": False,
        "retrieval_only_completion_diagnostic_added": True,
        "llm_verified_context_diagnostic_added": True,
        "combined_latency_target_seconds": 25.0,
        "live_add_modify_delete_probe_added": True,
        "live_lifecycle_run_by_installer": False,
        "fresh_retrieval_llm_benchmark_run_by_installer": False,
        "cleanup": cleanup_report,
        "chunk_size": 900,
        "chunk_overlap": 150,
        "minimum_retrieval_score": 0.55,
        "retrieval_top_k": {"vector": 10, "bm25": 10, "final": 3},
        "multi_query_alternative_count": 2,
        "models_changed": False,
        "production_qdrant_or_bm25_rebuild_performed": False,
        "simulation": SIMULATE,
        "next": (
            "Verify UI during one KB update, then run the v6.5.5 completion suite: "
            "retrieval-only gate -> LLM-from-verified-context -> latency; optionally include live KB lifecycle."
        ),
    }
    summary_path = outdir / f"v6.5.5_goal_completion_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    validation_json = outdir / "validation_latest.json"
    portability_json = root / "logs" / "kb_update_portability" / "kb_update_portability_latest.json"
    cleanup_report_path = cleanup_archive / "cleanup_report.json" if cleanup_archive else None

    result_zip = outdir / f"DocuBot_v6.5.5_Goal_Completion_UI_Diagnostics_Cleanup_Result_{stamp}.zip"
    with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        paths = [
            summary_path,
            marker,
            validation_json,
            root / "FINAL_COMPONENT_MANIFEST_v6.5.5.json",
            root / "cleanup_plan_v6.5.5.json",
            root / "docs" / "GOAL_COMPLETION_SMOKE_v6.5.5.txt",
            root / "docs" / "FINAL_PROJECT_STRUCTURE_v6.5.5.txt",
            portability_json,
            cleanup_report_path,
        ]
        for path in paths:
            if path and path.is_file():
                try:
                    arcname = path.relative_to(root).as_posix()
                except ValueError:
                    arcname = path.name
                archive.write(path, arcname)
        archive.writestr("python_environment_probe.txt", python_probe or "")
        archive.writestr("validator_console.txt", validation.stdout or "")
        archive.writestr("semantic_source_check.txt", semantic_source_check.stdout or "")
        archive.writestr("semantic_index_readiness.txt", semantic_ready.stdout or "")
        archive.writestr("kb_health_output.txt", kb_health.stdout or "")
        archive.writestr("kb_update_portability_console.txt", portability.stdout or "")

    result_hash = sha256_file(result_zip)
    sidecar = result_zip.with_suffix(result_zip.suffix + ".sha256.txt")
    sidecar.write_text(f"{result_hash}  {result_zip.name}\n", encoding="utf-8")

    print("[PASS] v6.5.5 installation completed.")
    print("[INFO] Cleanup archived exact legacy files:", cleanup_report["archived_count"])
    print("[INFO] Cleanup preserved hash-mismatched/user-modified files:", cleanup_report["skipped_modified_count"])
    print("[RESULT]", result_zip)
    print("[SHA256]", result_hash)
    print("[NEXT] Send the Result ZIP + SHA256 first. Runtime completion gates are intentionally separate from installation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
