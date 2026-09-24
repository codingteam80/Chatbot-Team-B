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
BASELINE_ALLOWED = json.loads((PATCH / "baseline_allowed_hashes.json").read_text(encoding="utf-8"))
BASELINE_MANIFESTS = json.loads((PATCH / "baseline_manifests.json").read_text(encoding="utf-8"))
SIMULATE = os.getenv("DOCUBOT_V654_SIMULATE", "").strip() == "1"
VERSION = "v6.5.4"
SEMANTIC_META_REL = "storage/option_c_qwen3_qdrant_v4/metadata/misra_semantic_rule_index_v1.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def locate_project(arg: str | None = None) -> Path | None:
    candidates = []
    if arg:
        candidates.append(Path(arg).expanduser())
    candidates += [PATCH.parent, PATCH.parent / "company-chatbot", PATCH.parent.parent, PATCH.parent.parent / "company-chatbot"]
    seen = set()
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
                python_exe, "-X", "utf8", "-u", "-c",
                (
                    "import sys,qdrant_client,streamlit,rank_bm25,numpy; "
                    "print(sys.executable); print('qdrant_client=OK'); "
                    "print('streamlit=OK'); print('rank_bm25=OK'); print('numpy=OK')"
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
            "A live DocuBot/Streamlit or KB-update process still owns logs\\kb_update\\kb_update.lock "
            f"(PID {pid}, source={existing.get('source', 'unknown')}). Close DocuBot/Streamlit and rerun."
        )
    try:
        lock_path.unlink(missing_ok=True)
        return True, f"Removed stale KB update lock from dead PID {pid or 'unknown'}."
    except Exception as error:
        return False, f"Could not remove stale KB update lock: {type(error).__name__}: {error}"


def baseline_ok(root: Path):
    problems = []
    matched = {}
    for rel, allowed in BASELINE_ALLOWED.items():
        path = root / rel
        if not path.is_file():
            problems.append(f"missing baseline: {rel}")
            continue
        actual = sha256_file(path).lower()
        allowed_l = [str(x).lower() for x in allowed]
        if actual not in allowed_l:
            problems.append(f"baseline mismatch: {rel} => {actual}")
        else:
            matched[rel] = actual

    manifest_matches = []
    for rel, wanted in BASELINE_MANIFESTS.items():
        path = root / rel
        if path.is_file() and sha256_file(path).lower() == str(wanted).lower():
            manifest_matches.append(rel)
    if not manifest_matches:
        problems.append("neither exact v6.5.3 nor exact v6.5.3.1 component manifest was found")
    return not problems, problems, matched, manifest_matches


def print_output(result):
    output = result.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n")


def restore(root: Path, history: Path, tracked: dict[str, bool]):
    for rel, existed in tracked.items():
        dst = root / rel
        backup = history / rel
        if existed and backup.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, dst)
        elif not existed:
            try:
                dst.unlink(missing_ok=True)
            except Exception:
                pass


def completed(output=""):
    return subprocess.CompletedProcess([], 0, stdout=output)


def main() -> int:
    root = locate_project(sys.argv[1] if len(sys.argv) > 1 else None)
    if root is None:
        print("[FAIL] DocuBot project not found. No files changed.")
        return 2

    print("=" * 112)
    print("DocuBot v6.5.4 - Incremental KB Lifecycle + Semantic Scope + Source Stability")
    print("=" * 112)
    print("[INFO] Restores intended Update Knowledge Base lifecycle: unchanged=SKIP, new/modified/deleted=PROCESS.")
    print("[INFO] Preserves v6.5.3.1 worker lock handoff and semantic latency diagnostics.")
    print("[INFO] Adds changed-file source snapshots and pre-commit source stability guards.")
    print("[INFO] Prevents unrelated office documents from forcing MISRA Rule-profile re-embedding.")
    print("[INFO] Chunking stays 900/150; threshold stays 0.55; models and Top-K stay unchanged.")
    print("[INFO] Installer does NOT rebuild production Qdrant/BM25.")

    safe, note = inspect_existing_kb_lock(root)
    if not safe:
        print("[BLOCKED]", note)
        print("[INFO] No project files were changed.")
        return 13
    print("[PASS]", note)

    payload_problems = []
    for rel, wanted in PAYLOAD_HASHES.items():
        path = PAYLOAD / rel
        if not path.is_file():
            payload_problems.append(f"missing payload: {rel}")
        elif sha256_file(path).lower() != str(wanted).lower():
            payload_problems.append(f"payload hash mismatch: {rel}")
    if payload_problems:
        print("[FAIL] Payload integrity failed. No files changed.")
        for item in payload_problems:
            print(" -", item)
        return 4
    print(f"[PASS] Payload integrity verified ({len(PAYLOAD_HASHES)} files).")

    ok, problems, matched, manifest_matches = baseline_ok(root)
    if not ok:
        print("[FAIL] Project is not a verified v6.5.3/v6.5.3.1 baseline. No files changed.")
        for item in problems:
            print(" -", item)
        return 3
    print("[PASS] Verified compatible v6.5.3/v6.5.3.1 baseline.")
    print("[INFO] Baseline manifest:", ", ".join(manifest_matches))

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
    history = root / "evidence" / "release_history" / f"pre_v6.5.4_{stamp}"
    history.mkdir(parents=True, exist_ok=True)

    tracked_rels = list(PAYLOAD_HASHES)
    tracked_rels += list(BASELINE_MANIFESTS)
    tracked_rels += [SEMANTIC_META_REL, "INCREMENTAL_KB_SEMANTIC_SCOPE_APPLIED_v6.5.4.txt"]
    tracked = {}
    for rel in dict.fromkeys(tracked_rels):
        src = root / rel
        existed = src.is_file()
        tracked[rel] = existed
        if existed:
            backup = history / rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, backup)

    try:
        for src in PAYLOAD.rglob("*"):
            if src.is_file():
                rel = src.relative_to(PAYLOAD)
                dst = root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    except Exception as error:
        print("[FAIL] Payload copy failed; restoring backup:", error)
        restore(root, history, tracked)
        return 6

    marker = root / "INCREMENTAL_KB_SEMANTIC_SCOPE_APPLIED_v6.5.4.txt"
    marker.write_text(
        "\n".join([
            "DocuBot v6.5.4 Incremental KB Lifecycle + Semantic Scope + Source Stability",
            "=" * 100,
            f"Applied: {datetime.now().isoformat(timespec='seconds')}",
            "Base: verified v6.5.3/v6.5.3.1 compatible baseline",
            "UNCHANGED -> SKIP existing vectors: YES",
            "NEW -> process only new file: YES",
            "MODIFIED -> replace only modified file: YES",
            "DELETED -> remove only deleted file: YES",
            "Full rebuild reserved for incompatible/missing index state: YES",
            "Changed-file source snapshot: YES",
            "Precommit source stability guard: YES",
            "Full-rebuild source stability guard: YES",
            "MISRA semantic signature scoped to Rule profiles: YES",
            "Non-MISRA change avoids MISRA profile re-embedding: YES",
            "Worker-only KB lock handoff preserved: YES",
            "Semantic latency diagnostics preserved: YES",
            "Chunk size/overlap changed: NO (900/150)",
            "Global retrieval threshold changed: NO (0.55)",
            "Vector/BM25/Final Top-K changed: NO (10/10/3)",
            "Models changed: NO",
            "Production Qdrant/BM25 rebuilt by installer: NO",
            f"Backup: {history}",
        ]) + "\n",
        encoding="utf-8",
    )

    env = {"DOCUBOT_V654_SIMULATE": "1"} if SIMULATE else None
    validation = run_cmd(root, [python_exe, "-X", "utf8", "-u", "-m", "scripts.validate_v6_5_4_incremental_kb_semantic_scope"], env=env)
    print_output(validation)
    if validation.returncode != 0:
        print("[FAIL] v6.5.4 validation failed. Restoring backup.")
        restore(root, history, tracked)
        return 7

    if SIMULATE:
        source_check = completed("SIMULATION: semantic source/profile check skipped.\n")
        semantic_migration = completed("SIMULATION: semantic metadata-only migration skipped.\n")
        semantic_ready = completed("SIMULATION: semantic readiness skipped.\n")
        kb_health = completed("SIMULATION: KB Health skipped.\n")
        portability = completed("SIMULATION: portability dry run skipped.\n")
    else:
        source_check = run_module(root, python_exe, "scripts.build_misra_semantic_rule_index", "--check")
        print_output(source_check)
        if source_check.returncode != 0:
            print("[FAIL] MISRA semantic source/profile check failed. Restoring backup.")
            restore(root, history, tracked)
            return 8

        semantic_migration = run_cmd(
            root,
            [python_exe, "-X", "utf8", "-u", "-c", (
                "import json,sys; "
                "from retrieval.semantic_rule_resolver import migrate_semantic_rule_index_signature_if_compatible as m; "
                "r=m(); print(json.dumps(r,indent=2)); "
                "sys.exit(0 if r.get('ready') else 4)"
            )],
        )
        print_output(semantic_migration)
        if semantic_migration.returncode != 0:
            print("[FAIL] Existing semantic Rule index cannot be migrated without model inference. No production KB rebuild was attempted. Restoring backup.")
            restore(root, history, tracked)
            return 9

        semantic_ready = run_cmd(
            root,
            [python_exe, "-X", "utf8", "-u", "-c", (
                "from retrieval.semantic_rule_resolver import semantic_rule_index_ready; "
                "import sys; ok=semantic_rule_index_ready(); print('semantic_rule_index_ready='+str(ok)); sys.exit(0 if ok else 3)"
            )],
        )
        print_output(semantic_ready)
        if semantic_ready.returncode != 0:
            print("[FAIL] Semantic Rule index readiness failed after metadata migration. Restoring backup.")
            restore(root, history, tracked)
            return 10

        kb_health = run_module(root, python_exe, "scripts.kb_health")
        print_output(kb_health)
        if kb_health.returncode != 0:
            print("[FAIL] KB Health failed. Restoring backup.")
            restore(root, history, tracked)
            return 11

        portability = run_module(root, python_exe, "scripts.test_kb_update_portability")
        print_output(portability)
        if portability.returncode != 0:
            print("[FAIL] KB update portability/incremental dry run failed. Restoring backup.")
            restore(root, history, tracked)
            return 12

    # Retire prior release manifests only after every gate passed.
    for rel in BASELINE_MANIFESTS:
        try:
            (root / rel).unlink(missing_ok=True)
        except Exception:
            pass

    outdir = root / "logs" / "incremental_kb_semantic_scope"
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": VERSION,
        "overall": "PASS",
        "base": "verified v6.5.3/v6.5.3.1 compatible baseline",
        "selected_project_python": python_exe,
        "validation_exit_code": validation.returncode,
        "semantic_source_check_exit_code": source_check.returncode,
        "semantic_signature_migration_exit_code": semantic_migration.returncode,
        "semantic_rule_index_ready": semantic_ready.returncode == 0,
        "kb_health_exit_code": kb_health.returncode,
        "kb_update_portability_dry_run_exit_code": portability.returncode,
        "chunk_size": 900,
        "chunk_overlap": 150,
        "minimum_retrieval_score": 0.55,
        "retrieval_top_k": {"vector": 10, "bm25": 10, "final": 3},
        "multi_query_alternative_count": 2,
        "kb_update_incremental_lifecycle": True,
        "unchanged_files_reembedded": False,
        "changed_file_source_snapshot": True,
        "precommit_source_stability_guard": True,
        "semantic_profile_scoped_signature": True,
        "non_misra_change_reembeds_misra_profiles": False,
        "kb_update_worker_lock_handoff_preserved": True,
        "semantic_latency_diagnostics_preserved": True,
        "models_changed": False,
        "production_qdrant_or_bm25_rebuild_performed": False,
        "simulation": SIMULATE,
        "next": "TEST UPDATE KNOWLEDGE BASE WITH ONE NEW/MODIFIED/DELETED FILE; UNCHANGED MISRA MUST BE SKIPPED.",
    }
    summary_path = outdir / f"v6.5.4_incremental_kb_semantic_scope_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    validation_json = outdir / "v6.5.4_incremental_kb_semantic_scope_validation_latest.json"
    portability_json = root / "logs" / "kb_update_portability" / "kb_update_portability_latest.json"
    result_zip = outdir / f"DocuBot_v6.5.4_Incremental_KB_Lifecycle_Semantic_Scope_Result_{stamp}.zip"
    with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in [
            summary_path,
            marker,
            validation_json,
            root / "FINAL_COMPONENT_MANIFEST_v6.5.4.json",
            root / "docs" / "INCREMENTAL_KB_SEMANTIC_SCOPE_SMOKE_v6.5.4.txt",
            portability_json,
        ]:
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("python_environment_probe.txt", python_probe or "")
        archive.writestr("validator_console.txt", validation.stdout or "")
        archive.writestr("semantic_source_check.txt", source_check.stdout or "")
        archive.writestr("semantic_signature_migration.txt", semantic_migration.stdout or "")
        archive.writestr("semantic_index_readiness.txt", semantic_ready.stdout or "")
        archive.writestr("kb_health_output.txt", kb_health.stdout or "")
        archive.writestr("kb_update_portability_console.txt", portability.stdout or "")

    result_hash = sha256_file(result_zip)
    sidecar = result_zip.with_suffix(result_zip.suffix + ".sha256.txt")
    sidecar.write_text(f"{result_hash}  {result_zip.name}\n", encoding="utf-8")

    print("[PASS] v6.5.4 installation completed.")
    print("[RESULT]", result_zip)
    print("[SHA256]", result_hash)
    print("[NEXT] Test Update Knowledge Base with one changed source. Unchanged MISRA must show as skipped and must not be re-embedded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
