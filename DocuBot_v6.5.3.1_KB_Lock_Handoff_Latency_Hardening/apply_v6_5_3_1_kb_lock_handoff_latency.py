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

PATCH = Path(__file__).resolve().parent
PAYLOAD = PATCH / "_payload"
PAYLOAD_HASHES = json.loads((PATCH / "payload_hashes.json").read_text(encoding="utf-8"))
BASELINE_HASHES = json.loads((PATCH / "baseline_hashes.json").read_text(encoding="utf-8"))
SIMULATE = os.getenv("DOCUBOT_V6531_SIMULATE", "").strip() == "1"
VERSION = "v6.5.3.1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def locate_project(arg: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if arg:
        candidates.append(Path(arg).expanduser())
    candidates.extend([
        PATCH.parent,
        PATCH.parent / "company-chatbot",
        PATCH.parent.parent,
        PATCH.parent.parent / "company-chatbot",
    ])
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
    attempts = []
    for python_exe in python_candidates(root):
        if SIMULATE and python_exe == sys.executable:
            return python_exe, "SIMULATION: project dependency probe bypassed.", attempts
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
                    "print(sys.executable); "
                    "print('qdrant_client=OK'); print('streamlit=OK'); "
                    "print('rank_bm25=OK'); print('numpy=OK')"
                ),
            ],
        )
        attempts.append({"python": python_exe, "returncode": probe.returncode, "output": probe.stdout.strip()})
        if probe.returncode == 0:
            return python_exe, probe.stdout.strip(), attempts
    return None, "", attempts


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
            )
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


def inspect_existing_kb_lock(root: Path) -> tuple[bool, str]:
    """Return (safe_to_continue, note). Never remove a lock owned by a live PID."""

    lock_path = root / "logs" / "kb_update" / "kb_update.lock"
    if not lock_path.is_file():
        return True, "No existing KB update lock."

    try:
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            existing = {}
    except Exception:
        existing = {}

    pid = int(existing.get("pid", 0) or 0)
    if pid_is_running(pid):
        return False, (
            "A live DocuBot/Streamlit or KB-update process still owns "
            f"logs\\kb_update\\kb_update.lock (PID {pid}, "
            f"source={existing.get('source', 'unknown')}). "
            "Close DocuBot/Streamlit and rerun this installer."
        )

    try:
        lock_path.unlink(missing_ok=True)
        return True, f"Removed stale KB update lock from dead PID {pid or 'unknown'}."
    except Exception as error:
        return False, f"Could not remove stale KB update lock: {type(error).__name__}: {error}"


def restore(
    root: Path,
    history: Path,
    applied_rels: list[str],
    old_manifest_was_present: bool,
) -> None:
    for rel in applied_rels:
        dst = root / rel
        src = history / rel
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        elif dst.exists():
            try:
                dst.unlink()
            except Exception:
                pass

    old_manifest = root / "FINAL_COMPONENT_MANIFEST_v6.5.3.json"
    old_backup = history / "FINAL_COMPONENT_MANIFEST_v6.5.3.json"
    if old_manifest_was_present and old_backup.is_file():
        shutil.copy2(old_backup, old_manifest)

    new_manifest = root / "FINAL_COMPONENT_MANIFEST_v6.5.3.1.json"
    if not (history / "FINAL_COMPONENT_MANIFEST_v6.5.3.1.json").is_file():
        try:
            new_manifest.unlink(missing_ok=True)
        except Exception:
            pass


def print_output(result) -> None:
    output = result.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n")


def completed(output: str = ""):
    return subprocess.CompletedProcess([], 0, stdout=output)


def main() -> int:
    root = locate_project(sys.argv[1] if len(sys.argv) > 1 else None)
    if root is None:
        print("[FAIL] DocuBot project not found. No files changed.")
        return 2

    print("=" * 106)
    print("DocuBot v6.5.3.1 - KB Update Worker Lock Handoff + Latency Decision Diagnostics")
    print("=" * 106)
    print("[INFO] Hotfixes the field-observed Update Knowledge Base parent/worker self-lock.")
    print("[INFO] Preserves v6.5.3 Yes/No, follow-up, semantic early-accept and prewarm behavior.")
    print("[INFO] Adds explicit single-query / MultiQuery latency-decision evidence.")
    print("[INFO] Chunking stays 900/150; threshold stays 0.55; models and Top-K stay unchanged.")
    print("[INFO] Installer never rebuilds production Qdrant/BM25.")

    safe, lock_note = inspect_existing_kb_lock(root)
    if not safe:
        print("[BLOCKED]", lock_note)
        print("[INFO] No project files were changed.")
        return 13
    print("[PASS]", lock_note)

    problems = []
    for rel, wanted in PAYLOAD_HASHES.items():
        path = PAYLOAD / rel
        if not path.is_file():
            problems.append(f"missing payload: {rel}")
        elif sha256_file(path).lower() != str(wanted).lower():
            problems.append(f"payload hash mismatch: {rel}")
    if problems:
        print("[FAIL] Payload integrity failed. No files changed.")
        for problem in problems:
            print(" -", problem)
        return 4
    print(f"[PASS] Payload integrity verified ({len(PAYLOAD_HASHES)} files).")

    problems = []
    for rel, wanted in BASELINE_HASHES.items():
        path = root / rel
        if not path.is_file():
            problems.append(f"missing baseline: {rel}")
        elif sha256_file(path).lower() != str(wanted).lower():
            problems.append(f"baseline mismatch: {rel}")
    if problems:
        print("[FAIL] Project is not the exact verified v6.5.3 baseline. No files changed.")
        for problem in problems:
            print(" -", problem)
        return 3
    print("[PASS] Exact v6.5.3 baseline verified.")

    python_exe, python_probe, attempts = choose_python(root)
    if not python_exe:
        print("[FAIL] No project Python with required DocuBot dependencies was found. No files changed.")
        for attempt in attempts:
            print(f" - {attempt['python']} => rc={attempt['returncode']}")
            if attempt.get("output"):
                print(attempt["output"])
        return 5
    print("[PASS] Project Python selected:", python_exe)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history = root / "evidence" / "release_history" / f"v6.5.3_pre_v6.5.3.1_{stamp}"
    history.mkdir(parents=True, exist_ok=True)

    applied_rels = list(PAYLOAD_HASHES.keys())
    for rel in applied_rels:
        current = root / rel
        if current.is_file():
            backup = history / rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(current, backup)

    old_manifest = root / "FINAL_COMPONENT_MANIFEST_v6.5.3.json"
    old_manifest_was_present = old_manifest.is_file()
    if old_manifest_was_present:
        shutil.copy2(old_manifest, history / old_manifest.name)

    try:
        for src in PAYLOAD.rglob("*"):
            if src.is_file():
                rel = src.relative_to(PAYLOAD)
                dst = root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    except Exception as exc:
        print("[FAIL] Payload copy failed; restoring v6.5.3 backup:", exc)
        restore(root, history, applied_rels, old_manifest_was_present)
        return 6

    print(f"[PASS] Applied {len(applied_rels)} payload files; backup: {history}")

    marker = root / "KB_UPDATE_LOCK_HANDOFF_LATENCY_APPLIED_v6.5.3.1.txt"
    marker.write_text(
        "\n".join([
            "DocuBot v6.5.3.1 KB Update Worker Lock Handoff + Latency Decision Diagnostics",
            "=" * 94,
            f"Applied: {datetime.now().isoformat(timespec='seconds')}",
            "Base: exact verified v6.5.3",
            "Chunk size/overlap changed: NO (900/150)",
            "Global retrieval threshold changed: NO (0.55)",
            "Vector/BM25/Final Top-K changed: NO (10/10/3)",
            "Models changed: NO",
            "Production Qdrant/BM25 rebuilt by installer: NO",
            "Worker-only KB update lock ownership: YES",
            "Legacy Streamlit parent self-lock handoff: YES",
            "Process-start identity guard for PID reuse: YES",
            "Live worker lock protection: YES",
            "Streamlit session double-click guard: YES",
            "v6.5.3 single-query early accept preserved: YES",
            "v6.5.3 MultiQuery fallback preserved: YES",
            "Semantic latency decision event: YES",
            "Single-query / MultiQuery / widened timing fields: YES",
            f"Backup: {history}",
        ]) + "\n",
        encoding="utf-8",
    )

    validation = run_module(root, python_exe, "scripts.validate_v6_5_3_1_kb_lock_handoff_latency")
    print_output(validation)
    if validation.returncode != 0:
        print("[FAIL] v6.5.3.1 validation failed. Restoring v6.5.3 code backup.")
        restore(root, history, applied_rels, old_manifest_was_present)
        return 7

    if SIMULATE:
        regression = completed("SIMULATION: v6.5.3 regression validator skipped.\n")
        source_check = completed("SIMULATION: semantic source/profile check skipped.\n")
        semantic_ready = completed("SIMULATION: live semantic Rule readiness skipped.\n")
        kb_health = completed("SIMULATION: live KB Health skipped.\n")
        portability = completed("SIMULATION: live project portability dry run skipped.\n")
    else:
        regression = run_module(root, python_exe, "scripts.validate_v6_5_3_answer_kb_update_portability")
        print_output(regression)
        if regression.returncode != 0:
            print("[FAIL] v6.5.3 regression validator failed. Restoring v6.5.3 code backup.")
            restore(root, history, applied_rels, old_manifest_was_present)
            return 8

        source_check = run_module(root, python_exe, "scripts.build_misra_semantic_rule_index", "--check")
        print_output(source_check)
        if source_check.returncode != 0:
            print("[FAIL] Semantic Rule source/profile check failed. Restoring v6.5.3 code backup.")
            restore(root, history, applied_rels, old_manifest_was_present)
            return 9

        semantic_ready = run_cmd(
            root,
            [
                python_exe,
                "-X",
                "utf8",
                "-u",
                "-c",
                (
                    "from retrieval.semantic_rule_resolver import semantic_rule_index_ready; "
                    "import sys; ok=semantic_rule_index_ready(); "
                    "print('semantic_rule_index_ready='+str(ok)); sys.exit(0 if ok else 3)"
                ),
            ],
        )
        print_output(semantic_ready)
        if semantic_ready.returncode != 0:
            print("[FAIL] Existing derived semantic Rule index is not ready. No KB rebuild will be attempted by this hotfix.")
            restore(root, history, applied_rels, old_manifest_was_present)
            return 10

        kb_health = run_module(root, python_exe, "scripts.kb_health")
        print_output(kb_health)
        if kb_health.returncode != 0:
            print("[FAIL] KB Health failed. Restoring v6.5.3 code backup.")
            restore(root, history, applied_rels, old_manifest_was_present)
            return 11

        portability = run_module(root, python_exe, "scripts.test_kb_update_portability")
        print_output(portability)
        if portability.returncode != 0:
            print("[FAIL] KB update lock-handoff/portability dry run failed. Restoring v6.5.3 code backup.")
            restore(root, history, applied_rels, old_manifest_was_present)
            return 12

    # Retire the old component manifest only after every gate is clean.
    if old_manifest.is_file():
        old_manifest.unlink()

    outdir = root / "logs" / "kb_lock_handoff_latency"
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": VERSION,
        "overall": "PASS",
        "base": "exact verified v6.5.3",
        "selected_project_python": python_exe,
        "validation_exit_code": validation.returncode,
        "v6_5_3_regression_exit_code": regression.returncode,
        "semantic_source_check_exit_code": source_check.returncode,
        "semantic_rule_index_ready": semantic_ready.returncode == 0,
        "kb_health_exit_code": kb_health.returncode,
        "kb_update_portability_dry_run_exit_code": portability.returncode,
        "chunk_size": 900,
        "chunk_overlap": 150,
        "minimum_retrieval_score": 0.55,
        "retrieval_top_k": {"vector": 10, "bm25": 10, "final": 3},
        "multi_query_alternative_count": 2,
        "kb_update_worker_only_lock_owner": True,
        "kb_update_legacy_parent_self_lock_handoff": True,
        "kb_update_process_identity_guard": True,
        "kb_update_live_worker_lock_protection": True,
        "kb_update_streamlit_double_click_guard": True,
        "single_query_semantic_bge_early_accept_preserved": True,
        "semantic_latency_decision_diagnostics": True,
        "semantic_acceptance_thresholds_changed": False,
        "models_changed": False,
        "production_qdrant_or_bm25_rebuild_performed": False,
        "simulation": SIMULATE,
        "next": "START DOCUBOT, TEST UPDATE KNOWLEDGE BASE ONCE, THEN SEND KB UPDATE RESULT JSON/CONSOLE IF ANY.",
    }
    summary_path = outdir / f"v6.5.3.1_kb_lock_handoff_latency_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    validation_json = outdir / "v6.5.3.1_kb_lock_handoff_latency_validation_latest.json"
    portability_json = root / "logs" / "kb_update_portability" / "kb_update_portability_latest.json"
    result_zip = outdir / f"DocuBot_v6.5.3.1_KB_Lock_Handoff_Latency_Hardening_Result_{stamp}.zip"
    with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in [
            summary_path,
            marker,
            validation_json,
            root / "FINAL_COMPONENT_MANIFEST_v6.5.3.1.json",
            root / "docs" / "KB_LOCK_HANDOFF_LATENCY_SMOKE_v6.5.3.1.txt",
            portability_json,
        ]:
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("python_environment_probe.txt", python_probe or "")
        archive.writestr("validator_console.txt", validation.stdout or "")
        archive.writestr("v6.5.3_regression_console.txt", regression.stdout or "")
        archive.writestr("semantic_source_check.txt", source_check.stdout or "")
        archive.writestr("semantic_index_readiness.txt", semantic_ready.stdout or "")
        archive.writestr("kb_health_output.txt", kb_health.stdout or "")
        archive.writestr("kb_update_portability_console.txt", portability.stdout or "")

    result_hash = sha256_file(result_zip)
    sidecar = result_zip.with_suffix(result_zip.suffix + ".sha256.txt")
    sidecar.write_text(f"{result_hash}  {result_zip.name}\n", encoding="utf-8")

    print("[PASS] v6.5.3.1 installation completed.")
    print("[RESULT]", result_zip)
    print("[SHA256]", result_hash)
    print("[NEXT] Start DocuBot and test Update Knowledge Base ONCE. If it does not PASS, send the new kb_update_result JSON + console log.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
