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
BASELINE = json.loads((PATCH / "baseline_hashes.json").read_text(encoding="utf-8"))
PAYLOAD_HASHES = json.loads((PATCH / "payload_hashes.json").read_text(encoding="utf-8"))
VERSION = "v6.5.5.1"
SIMULATE = os.getenv("DOCUBOT_V6551_SIMULATE", "").strip() == "1"
FORCE_FAIL = os.getenv("DOCUBOT_V6551_FORCE_VALIDATION_FAIL", "").strip() == "1"
RESULT_REL = "logs/v6_5_5_1_notification_ux"
MARKER_REL = "KB_UPDATE_NOTIFICATION_UX_HOTFIX_APPLIED_v6.5.5.1.txt"
OLD_MANIFEST_REL = "FINAL_COMPONENT_MANIFEST_v6.5.5.json"
NEW_MANIFEST_REL = "FINAL_COMPONENT_MANIFEST_v6.5.5.1.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def locate_project(arg: str | None) -> Path | None:
    candidates = []
    if arg:
        candidates.append(Path(arg).expanduser())
    candidates += [PATCH.parent, PATCH.parent / "company-chatbot", Path.cwd(), Path.cwd() / "company-chatbot"]
    seen = set()
    for c in candidates:
        try:
            r = c.resolve()
        except Exception:
            continue
        k = str(r).casefold()
        if k in seen:
            continue
        seen.add(k)
        if (r / "run.py").is_file() and (r / "config" / "settings.py").is_file():
            return r
    return None


def run_cmd(root: Path, args: list[str], env=None):
    e = os.environ.copy()
    if env:
        e.update(env)
    return subprocess.run(args, cwd=str(root), env=e, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def python_candidates(root: Path):
    for p in (root / "venv" / "Scripts" / "python.exe", root / ".venv" / "Scripts" / "python.exe", root / "env" / "Scripts" / "python.exe"):
        if p.is_file():
            yield str(p)
    if SIMULATE:
        yield sys.executable


def choose_python(root: Path):
    attempts = []
    for exe in python_candidates(root):
        if SIMULATE and exe == sys.executable:
            return exe, "SIMULATION: dependency probe bypassed.", attempts
        r = run_cmd(root, [exe, "-X", "utf8", "-u", "-c", "import sys,streamlit,qdrant_client,rank_bm25,numpy; print(sys.executable); print('dependencies=OK')"])
        attempts.append((exe, r.returncode, r.stdout or ""))
        if r.returncode == 0:
            return exe, r.stdout or "", attempts
    return None, "", attempts


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
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


def inspect_lock(root: Path):
    p = root / "logs" / "kb_update" / "kb_update.lock"
    if not p.is_file():
        return True, "No active KB update lock."
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        pid = int((obj or {}).get("pid", 0) or 0)
    except Exception:
        obj, pid = {}, 0
    if pid_is_running(pid):
        return False, f"Live KB update lock detected (PID {pid}, source={obj.get('source','unknown')}). Stop DocuBot/update first."
    try:
        p.unlink(missing_ok=True)
        return True, f"Removed stale KB update lock from dead PID {pid or 'unknown'}."
    except Exception as e:
        return False, f"Could not remove stale KB update lock: {type(e).__name__}: {e}"


def verify_payload():
    problems = []
    actual = {p.relative_to(PAYLOAD).as_posix() for p in PAYLOAD.rglob("*") if p.is_file()}
    expected = set(PAYLOAD_HASHES)
    for rel, wanted in PAYLOAD_HASHES.items():
        p = PAYLOAD / rel
        if not p.is_file():
            problems.append(f"missing payload: {rel}")
        elif sha256_file(p).lower() != str(wanted).lower():
            problems.append(f"payload hash mismatch: {rel}")
    for rel in sorted(actual - expected):
        problems.append(f"unmanifested payload: {rel}")
    for rel in sorted(expected - actual):
        problems.append(f"manifested payload absent: {rel}")
    return not problems, problems


def verify_baseline(root: Path):
    problems = []
    for rel, wanted in (BASELINE.get("required_existing_sha256") or {}).items():
        p = root / rel
        if not p.is_file():
            problems.append(f"missing v6.5.5 baseline file: {rel}")
        else:
            actual = sha256_file(p).lower()
            if actual != str(wanted).lower():
                problems.append(f"baseline mismatch: {rel} => {actual}")
    for rel in BASELINE.get("required_absent") or []:
        if (root / rel).exists():
            problems.append(f"expected absent on clean v6.5.5 baseline: {rel}")
    return not problems, problems


def backup(root: Path, history: Path, rels: list[str]):
    state = {}
    for rel in rels:
        src = root / rel
        state[rel] = src.is_file()
        if src.is_file():
            dst = history / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return state


def restore(root: Path, history: Path, state: dict[str, bool]):
    for rel, existed in state.items():
        dst = root / rel
        src = history / rel
        if existed and src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        elif not existed and dst.exists():
            if dst.is_file() or dst.is_symlink():
                dst.unlink(missing_ok=True)


def fail_restore(root, history, state, message, code):
    print("[FAIL]", message)
    print("[ROLLBACK] Restoring exact v6.5.5 files...")
    restore(root, history, state)
    print("[ROLLBACK] Complete.")
    return code


def main():
    root = locate_project(sys.argv[1] if len(sys.argv) > 1 else None)
    if root is None:
        print("[FAIL] DocuBot project not found. No files changed.")
        return 2

    print("=" * 104)
    print("DocuBot v6.5.5.1 - KB Update Notification UX Hotfix")
    print("=" * 104)
    print("[INFO] Scope: success toast only; technical KB details stay in logs.")
    print("[INFO] Existing v6.5.5 incremental KB/retrieval architecture is not changed.")
    print("[INFO] Installer does NOT rebuild Qdrant/BM25.")

    safe, note = inspect_lock(root)
    if not safe:
        print("[BLOCKED]", note)
        return 13
    print("[PASS]", note)

    ok, problems = verify_payload()
    if not ok:
        print("[FAIL] Payload integrity failed. No files changed.")
        for x in problems: print(" -", x)
        return 4
    print(f"[PASS] Payload integrity verified ({len(PAYLOAD_HASHES)} files).")

    ok, problems = verify_baseline(root)
    if not ok:
        print("[FAIL] Project is not the exact supported v6.5.5 baseline. No files changed.")
        for x in problems: print(" -", x)
        return 3
    print(f"[PASS] Exact v6.5.5 baseline verified ({len(BASELINE['required_existing_sha256'])} covered files).")

    py, probe, attempts = choose_python(root)
    if not py:
        print("[FAIL] Project Python/dependencies unavailable. No files changed.")
        for a in attempts: print(a[0], a[1], a[2])
        return 5
    print("[PASS] Project Python selected:", py)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history = root / "evidence" / "release_history" / f"pre_v6.5.5.1_{stamp}"
    history.mkdir(parents=True, exist_ok=True)
    tracked = list(PAYLOAD_HASHES) + [OLD_MANIFEST_REL, MARKER_REL]
    state = backup(root, history, list(dict.fromkeys(tracked)))

    try:
        for src in PAYLOAD.rglob("*"):
            if src.is_file():
                rel = src.relative_to(PAYLOAD)
                dst = root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        # Keep only the current root component manifest. The old exact manifest
        # is retained in the rollback/release-history backup above.
        (root / OLD_MANIFEST_REL).unlink(missing_ok=True)
        marker = root / MARKER_REL
        marker.write_text(
            "\n".join([
                "DocuBot v6.5.5.1 KB Update Notification UX Hotfix",
                "=" * 80,
                f"Applied: {datetime.now().isoformat(timespec='seconds')}",
                "Base: exact verified v6.5.5",
                "Success toast simplified: YES",
                "Success toast counts: Added / Updated / Deleted",
                "Technical KB details kept in logs: YES",
                "Incremental KB lifecycle changed: NO",
                "Retrieval/LLM/latency diagnostics changed: NO",
                "Models/config/chunking/Top-K changed: NO",
                "Production Qdrant/BM25 rebuilt by installer: NO",
                f"Rollback backup: {history}",
            ]) + "\n", encoding="utf-8"
        )
    except Exception as e:
        return fail_restore(root, history, state, f"Payload copy failed: {type(e).__name__}: {e}", 6)

    validation = run_cmd(root, [py, "-X", "utf8", "-u", "scripts/validate_v6_5_5_1_kb_notification_ux.py"])
    print(validation.stdout or "", end="")
    if FORCE_FAIL:
        validation = subprocess.CompletedProcess(validation.args, 99, stdout=(validation.stdout or "") + "\nFORCED TEST FAILURE\n")
    if validation.returncode != 0:
        return fail_restore(root, history, state, "v6.5.5.1 notification UX validation failed.", 7)

    if SIMULATE:
        kb = subprocess.CompletedProcess([], 0, stdout="SIMULATION: read-only KB Health skipped.\n")
    else:
        kb = run_cmd(root, [py, "-X", "utf8", "-u", "-m", "scripts.kb_health"])
        print(kb.stdout or "", end="")
        if kb.returncode != 0:
            return fail_restore(root, history, state, "Read-only KB Health failed.", 8)

    # Exact post-copy payload check before declaring PASS.
    post = []
    for rel, wanted in PAYLOAD_HASHES.items():
        p = root / rel
        if not p.is_file() or sha256_file(p).lower() != str(wanted).lower():
            post.append(rel)
    if post:
        return fail_restore(root, history, state, f"Post-install payload integrity failed: {post}", 9)

    outdir = root / RESULT_REL
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": VERSION,
        "overall": "PASS",
        "base": "exact verified v6.5.5",
        "selected_project_python": py,
        "validation_exit_code": validation.returncode,
        "kb_health_exit_code": kb.returncode,
        "success_notification": "Knowledge Base is successfully updated. Added/Updated/Deleted counts only.",
        "technical_details_hidden_from_toast": True,
        "technical_details_preserved_in_logs": True,
        "incremental_kb_lifecycle_changed": False,
        "retrieval_pipeline_changed": False,
        "models_changed": False,
        "chunk_size": 900,
        "chunk_overlap": 150,
        "minimum_retrieval_score": 0.55,
        "retrieval_top_k": {"vector": 10, "bm25": 10, "final": 3},
        "multi_query_alternative_count": 2,
        "production_qdrant_or_bm25_rebuild_performed": False,
        "simulation": SIMULATE,
        "next": "Restart DocuBot and visually verify one successful KB update toast, then continue v6.5.5 completion diagnostics.",
    }
    summary_path = outdir / f"v6.5.5.1_notification_ux_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    result_zip = outdir / f"DocuBot_v6.5.5.1_KB_Update_Notification_UX_Hotfix_Result_{stamp}.zip"
    with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in (summary_path, root / MARKER_REL, root / NEW_MANIFEST_REL, root / "logs" / "v6_5_5_1_notification_ux" / "validation_latest.json", root / "docs" / "KB_UPDATE_NOTIFICATION_UX_SMOKE_v6.5.5.1.txt"):
            if p.is_file():
                try: arc = p.relative_to(root).as_posix()
                except ValueError: arc = p.name
                z.write(p, arc)
        z.writestr("python_environment_probe.txt", probe or "")
        z.writestr("validator_console.txt", validation.stdout or "")
        z.writestr("kb_health_output.txt", kb.stdout or "")

    digest = sha256_file(result_zip)
    side = result_zip.with_suffix(result_zip.suffix + ".sha256.txt")
    side.write_text(f"{digest}  {result_zip.name}\n", encoding="utf-8")
    print("[PASS] v6.5.5.1 notification UX hotfix completed.")
    print("[RESULT]", result_zip)
    print("[SHA256]", digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
