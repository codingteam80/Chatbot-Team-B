from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.kb_health import run_health_check
from scripts.smart_build import get_kb_update_preflight, get_update_plan, smart_build

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "kb_update"
LOCK_FILE = LOG_DIR / "kb_update.lock"


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


def _read_lock() -> dict[str, Any]:
    try:
        value = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


@contextmanager
def _update_lock(source: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_FILE.exists():
        existing = _read_lock()
        pid = int(existing.get("pid", 0) or 0)
        if not _pid_is_running(pid):
            try:
                LOCK_FILE.unlink()
            except OSError:
                pass

    payload = {
        "pid": os.getpid(),
        "source": source,
        "created": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
    }
    try:
        fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        existing = _read_lock()
        raise RuntimeError(
            "Another DocuBot knowledge-base update is already running. "
            f"Lock owner: PID {existing.get('pid', 'unknown')} from {existing.get('source', 'unknown')}."
        ) from error

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        yield payload
    finally:
        try:
            current = _read_lock()
            if int(current.get("pid", 0) or 0) == os.getpid():
                LOCK_FILE.unlink(missing_ok=True)
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


def run_kb_update_worker(*, source: str = "manual", report_path: Path | None = None) -> dict[str, Any]:
    """Run the same guarded KB update workflow for web and BAT entry points."""

    started = time.perf_counter()
    report: dict[str, Any] = {
        "version": "v6.5.3",
        "source": source,
        "created": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(ROOT),
        "python_executable": sys.executable,
        "overall": "FAIL",
        "stage": "initializing",
        "message": "",
    }

    try:
        with _update_lock(source):
            plan = get_update_plan()
            report["plan"] = {
                "mode": plan.get("mode"),
                "reason": plan.get("reason"),
                "changes": _json_safe(plan.get("changes") or {}),
                "document_count": len(plan.get("documents") or []),
            }

            report["stage"] = "preflight"
            preflight = get_kb_update_preflight(plan.get("documents") or [])
            report["preflight"] = _json_safe(preflight)
            if not preflight.get("ok"):
                report["message"] = "KB update preflight failed; the previous working index was not changed."
                report["overall"] = "BLOCKED"
                return report

            report["stage"] = "build"
            if plan.get("mode") == "noop":
                build_ok = True
                report["build_action"] = "noop"
            else:
                build_ok = bool(smart_build())
                report["build_action"] = "transactional_full_rebuild"
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
    """Launch an isolated update worker using the exact interpreter serving DocuBot."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = LOG_DIR / f"kb_update_result_{stamp}_{os.getpid()}.json"
    console_path = LOG_DIR / f"kb_update_console_{stamp}_{os.getpid()}.txt"

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
    ]
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
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
                "version": "v6.5.3",
                "source": source,
                "overall": "FAIL",
                "stage": "worker_exit",
                "message": "The KB update worker exited without producing its result report.",
            }
        report["worker_exit_code"] = int(proc.returncode)
        report["console_log"] = str(console_path)
        report["launcher_seconds"] = round(time.perf_counter() - started, 4)
        return report
    except subprocess.TimeoutExpired as error:
        console = (error.stdout or "") if isinstance(error.stdout, str) else ""
        console_path.write_text(console, encoding="utf-8")
        return {
            "version": "v6.5.3",
            "source": source,
            "overall": "FAIL",
            "stage": "timeout",
            "message": f"Knowledge-base update exceeded {timeout_seconds} seconds and was stopped by the launcher.",
            "console_log": str(console_path),
        }
    except Exception as error:
        return {
            "version": "v6.5.3",
            "source": source,
            "overall": "FAIL",
            "stage": "launcher",
            "message": f"{type(error).__name__}: {error}",
            "console_log": str(console_path),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="DocuBot guarded knowledge-base update runner")
    parser.add_argument("--worker", action="store_true", help="run update in this process")
    parser.add_argument("--source", default="manual")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    report_path = Path(args.report).resolve() if args.report else None
    if args.worker:
        report = run_kb_update_worker(source=args.source, report_path=report_path)
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
