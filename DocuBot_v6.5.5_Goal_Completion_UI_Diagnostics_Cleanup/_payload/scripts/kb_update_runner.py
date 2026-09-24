from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.kb_health import run_health_check
from scripts.smart_build import (
    get_kb_update_preflight,
    get_last_build_result,
    get_update_plan,
    smart_build,
)

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "kb_update"
LOCK_FILE = LOG_DIR / "kb_update.lock"
VERSION = "v6.5.5"


def _json_safe(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def _pid_is_running(pid: int) -> bool:
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


def _read_lock(lock_path: Path = LOCK_FILE) -> dict[str, Any]:
    try:
        value = json.loads(lock_path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _process_start_token(pid: int) -> str:
    """Best-effort process creation identity used to reject PID-reuse stale locks."""

    if pid <= 0:
        return ""

    if os.name == "nt":
        try:
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

            class FILETIME(ctypes.Structure):
                _fields_ = [
                    ("dwLowDateTime", ctypes.c_uint32),
                    ("dwHighDateTime", ctypes.c_uint32),
                ]

            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
            )
            if not handle:
                return ""
            try:
                created = FILETIME()
                exited = FILETIME()
                kernel = FILETIME()
                user = FILETIME()
                ok = ctypes.windll.kernel32.GetProcessTimes(
                    handle,
                    ctypes.byref(created),
                    ctypes.byref(exited),
                    ctypes.byref(kernel),
                    ctypes.byref(user),
                )
                if not ok:
                    return ""
                value = (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)
                return str(value)
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            return ""

    try:
        stat_path = Path(f"/proc/{int(pid)}/stat")
        text = stat_path.read_text(encoding="utf-8", errors="replace")
        close = text.rfind(")")
        if close < 0:
            return ""
        fields = text[close + 2 :].split()
        # /proc/<pid>/stat starttime is field 22. fields[0] is field 3 (state).
        return str(fields[19]) if len(fields) > 19 else ""
    except Exception:
        return ""


def _lock_owner_alive(existing: dict[str, Any]) -> bool:
    pid = int(existing.get("pid", 0) or 0)
    if not _pid_is_running(pid):
        return False

    expected = str(existing.get("process_start_token", "") or "").strip()
    if not expected:
        # Legacy v6.5.3 lock. Treat a running PID as live unless the launcher
        # can prove it is its own parent/self-lock during handoff.
        return True

    actual = _process_start_token(pid)
    if not actual:
        return True
    return actual == expected


def _prepare_launcher_lock_handoff(
    *,
    source: str,
    parent_pid: int,
    lock_path: Path = LOCK_FILE,
) -> dict[str, Any]:
    """Prepare one safe parent->worker lock handoff without owning the worker lock.

    The Streamlit/BAT launcher never acquires the cross-process update lock.
    It may only remove:
      1) a dead/stale owner; or
      2) the verified legacy v6.5.3 parent self-lock whose PID is this launcher.
    Any other live owner blocks the new launch.
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if not lock_path.exists():
        return {"action": "none", "lock_path": str(lock_path)}

    existing = _read_lock(lock_path)
    pid = int(existing.get("pid", 0) or 0)
    existing_source = str(existing.get("source", "") or "")
    owner_role = str(existing.get("owner_role", "") or "")

    if not _lock_owner_alive(existing):
        try:
            lock_path.unlink(missing_ok=True)
            return {
                "action": "removed_stale_lock",
                "lock_path": str(lock_path),
                "previous": existing,
            }
        except Exception as error:
            return {
                "action": "blocked",
                "lock_path": str(lock_path),
                "previous": existing,
                "message": f"Could not remove stale KB update lock: {type(error).__name__}: {error}",
            }

    # v6.5.3 field failure: a lock could claim the live Streamlit PID and
    # source=streamlit_button before the child worker tried to acquire it.
    # The launcher itself is not an update worker, so this exact legacy
    # self-lock is safe to retire once during handoff.
    legacy_parent_self_lock = bool(
        pid == int(parent_pid)
        and existing_source == str(source or "")
        and owner_role in {"", "launcher", "streamlit_parent", "legacy_parent"}
    )
    if legacy_parent_self_lock:
        try:
            lock_path.unlink(missing_ok=True)
            return {
                "action": "removed_legacy_parent_self_lock",
                "lock_path": str(lock_path),
                "previous": existing,
            }
        except Exception as error:
            return {
                "action": "blocked",
                "lock_path": str(lock_path),
                "previous": existing,
                "message": f"Could not hand off legacy parent KB update lock: {type(error).__name__}: {error}",
            }

    return {
        "action": "blocked",
        "lock_path": str(lock_path),
        "previous": existing,
        "message": (
            "Another DocuBot knowledge-base update is already running. "
            f"Lock owner: PID {existing.get('pid', 'unknown')} "
            f"from {existing.get('source', 'unknown')} "
            f"({existing.get('owner_role', 'legacy/unknown')})."
        ),
    }


@contextmanager
def _update_lock(
    source: str,
    *,
    run_id: str = "",
    parent_pid: int = 0,
    lock_path: Path = LOCK_FILE,
):
    """Acquire the sole cross-process lock. Only the worker calls this."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)

    if lock_path.exists():
        existing = _read_lock(lock_path)
        if not _lock_owner_alive(existing):
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass

    payload = {
        "pid": os.getpid(),
        "source": source,
        "owner_role": "worker",
        "run_id": str(run_id or ""),
        "parent_pid": int(parent_pid or 0),
        "created": datetime.now().isoformat(timespec="seconds"),
        "created_epoch": time.time(),
        "python": sys.executable,
        "project_root": str(ROOT),
        "process_start_token": _process_start_token(os.getpid()),
    }
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        existing = _read_lock(lock_path)
        raise RuntimeError(
            "Another DocuBot knowledge-base update is already running. "
            f"Lock owner: PID {existing.get('pid', 'unknown')} "
            f"from {existing.get('source', 'unknown')} "
            f"({existing.get('owner_role', 'legacy/unknown')})."
        ) from error

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        yield payload
    finally:
        try:
            current = _read_lock(lock_path)
            same_owner = (
                int(current.get("pid", 0) or 0) == os.getpid()
                and str(current.get("run_id", "") or "") == str(run_id or "")
            )
            if same_owner:
                lock_path.unlink(missing_ok=True)
        except Exception:
            pass


def _write_report(report: dict[str, Any], report_path: Path | None = None) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if report_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = LOG_DIR / f"kb_update_result_{stamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report_path


def run_kb_update_worker(
    *,
    source: str = "manual",
    report_path: Path | None = None,
    run_id: str = "",
    parent_pid: int = 0,
) -> dict[str, Any]:
    """Run the same guarded KB update workflow for web and BAT entry points."""

    started = time.perf_counter()
    report: dict[str, Any] = {
        "version": VERSION,
        "source": source,
        "created": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(ROOT),
        "python_executable": sys.executable,
        "run_id": str(run_id or ""),
        "parent_pid": int(parent_pid or 0),
        "lock_owner_role": "worker",
        "overall": "FAIL",
        "stage": "initializing",
        "message": "",
    }

    try:
        with _update_lock(source, run_id=run_id, parent_pid=parent_pid):
            plan = get_update_plan()
            report["plan"] = {
                "mode": plan.get("mode"),
                "reason": plan.get("reason"),
                "changes": _json_safe(plan.get("changes") or {}),
                "document_count": len(plan.get("documents") or []),
            }

            report["stage"] = "preflight"
            changes = plan.get("changes") or {}
            requires_embedding = bool(
                plan.get("mode") == "full_rebuild"
                or changes.get("added")
                or changes.get("updated")
            )
            preflight = get_kb_update_preflight(
                plan.get("documents") or [],
                requires_embedding=requires_embedding,
            )
            report["preflight"] = _json_safe(preflight)
            if not preflight.get("ok"):
                report["message"] = "KB update preflight failed; the previous working index was not changed."
                report["overall"] = "BLOCKED"
                return report

            report["stage"] = "build"
            if plan.get("mode") == "noop":
                build_ok = True
                report["build_action"] = "noop"
                report["build_details"] = {"mode": "noop", "ok": True}
            else:
                build_ok = bool(smart_build())
                if plan.get("mode") == "incremental":
                    report["build_action"] = "transactional_incremental"
                else:
                    report["build_action"] = "transactional_full_rebuild"
                report["build_details"] = _json_safe(get_last_build_result())
            report["build_ok"] = build_ok
            if not build_ok:
                report["message"] = "Transactional KB update failed; rollback/preservation logic kept the previous working index where possible."
                return report

            # The Rule-level semantic index is derived from BM25. Rebuild it only
            # when the corpus signature changed; production Qdrant/BM25 are never
            # rebuilt by this step.
            report["stage"] = "semantic_index"
            try:
                from retrieval.semantic_rule_resolver import build_semantic_rule_index

                semantic = build_semantic_rule_index(force=False)
                report["semantic_rule_index"] = _json_safe(semantic)
                if not semantic.get("ready"):
                    report["message"] = "KB was built but the derived semantic Rule index is not ready."
                    return report
            except Exception as error:
                report["semantic_rule_index"] = {
                    "ready": False,
                    "error": f"{type(error).__name__}: {error}",
                }
                report["message"] = "KB was built but semantic Rule-index verification failed."
                return report

            report["stage"] = "kb_health"
            health_rc = int(run_health_check())
            report["kb_health_exit_code"] = health_rc
            if health_rc != 0:
                report["message"] = "Post-update KB Health did not pass. Review logs/kb_health and logs/kb_update before using the new source changes."
                return report

            post_plan = get_update_plan()
            report["post_plan_mode"] = post_plan.get("mode")
            report["post_plan_reason"] = post_plan.get("reason")
            if post_plan.get("mode") != "noop":
                report["message"] = "Post-update verification still detects an incomplete knowledge-base state."
                return report

            report["stage"] = "complete"
            report["overall"] = "PASS"
            report["message"] = "Knowledge base, BM25/Qdrant state, semantic Rule index, and KB Health all passed."
            return report

    except Exception as error:
        report["stage"] = report.get("stage") or "unexpected"
        report["message"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        return report
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        path = _write_report(report, report_path=report_path)
        report["report_path"] = str(path)
        # Rewrite once so report_path is also inside the file itself.
        _write_report(report, report_path=path)


def launch_kb_update_subprocess(*, source: str = "web", timeout_seconds: int = 7200) -> dict[str, Any]:
    """Launch the sole lock-owning update worker with an explicit handoff."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_pid = os.getpid()
    run_id = uuid.uuid4().hex[:16]
    report_path = LOG_DIR / f"kb_update_result_{stamp}_{parent_pid}.json"
    console_path = LOG_DIR / f"kb_update_console_{stamp}_{parent_pid}.txt"

    handoff = _prepare_launcher_lock_handoff(
        source=source,
        parent_pid=parent_pid,
    )
    if handoff.get("action") == "blocked":
        return {
            "version": VERSION,
            "source": source,
            "overall": "BLOCKED",
            "stage": "lock_handoff",
            "message": str(handoff.get("message") or "A live KB update lock already exists."),
            "launcher_lock_handoff": handoff,
        }

    cmd = [
        sys.executable,
        "-X",
        "utf8",
        "-u",
        "-m",
        "scripts.kb_update_runner",
        "--worker",
        "--source",
        source,
        "--report",
        str(report_path),
        "--run-id",
        run_id,
        "--parent-pid",
        str(parent_pid),
    ]
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env["DOCUBOT_KB_UPDATE_RUN_ID"] = run_id
    env["DOCUBOT_KB_UPDATE_PARENT_PID"] = str(parent_pid)
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(60, int(timeout_seconds)),
            check=False,
        )
        console = proc.stdout or ""
        console_path.write_text(console, encoding="utf-8")
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = {
                "version": VERSION,
                "source": source,
                "overall": "FAIL",
                "stage": "worker_exit",
                "message": "The KB update worker exited without producing its result report.",
            }
        report["worker_exit_code"] = int(proc.returncode)
        report["console_log"] = str(console_path)
        report["launcher_seconds"] = round(time.perf_counter() - started, 4)
        report["launcher_lock_handoff"] = handoff
        report["run_id"] = run_id
        report["parent_pid"] = parent_pid
        return report
    except subprocess.TimeoutExpired as error:
        console = (error.stdout or "") if isinstance(error.stdout, str) else ""
        console_path.write_text(console, encoding="utf-8")
        return {
            "version": VERSION,
            "source": source,
            "overall": "FAIL",
            "stage": "timeout",
            "message": f"Knowledge-base update exceeded {timeout_seconds} seconds and was stopped by the launcher.",
            "console_log": str(console_path),
            "launcher_lock_handoff": handoff,
            "run_id": run_id,
            "parent_pid": parent_pid,
        }
    except Exception as error:
        return {
            "version": VERSION,
            "source": source,
            "overall": "FAIL",
            "stage": "launcher",
            "message": f"{type(error).__name__}: {error}",
            "console_log": str(console_path),
            "launcher_lock_handoff": handoff,
            "run_id": run_id,
            "parent_pid": parent_pid,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="DocuBot guarded knowledge-base update runner")
    parser.add_argument("--worker", action="store_true", help="run update in this process")
    parser.add_argument("--source", default="manual")
    parser.add_argument("--report", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()

    report_path = Path(args.report).resolve() if args.report else None
    if args.worker:
        report = run_kb_update_worker(source=args.source, report_path=report_path, run_id=args.run_id, parent_pid=args.parent_pid)
    else:
        report = launch_kb_update_subprocess(source=args.source)

    print("DocuBot Knowledge Base Update")
    print("=" * 72)
    print("Overall :", report.get("overall"))
    print("Stage   :", report.get("stage"))
    print("Message :", report.get("message"))
    if report.get("report_path"):
        print("Report  :", report.get("report_path"))
    if report.get("console_log"):
        print("Console :", report.get("console_log"))
    return 0 if report.get("overall") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
