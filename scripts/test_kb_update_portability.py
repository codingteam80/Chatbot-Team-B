from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from scripts.kb_update_runner import _update_lock
from scripts.smart_build import get_kb_update_preflight, get_update_plan

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "logs" / "kb_update_portability"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "kb_update_portability_latest.json"


def main() -> int:
    checks = []

    def check(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail or "")})

    check("Project root resolved", (ROOT / "run.py").is_file(), ROOT)
    check("Current Python exists", Path(sys.executable).is_file(), sys.executable)

    normalized_python = str(Path(sys.executable).resolve()).casefold()
    normalized_root = str(ROOT.resolve()).casefold()
    project_python = normalized_python.startswith(normalized_root + os.sep.casefold())
    check("Running under project-local Python", project_python, sys.executable)

    child = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-c",
            (
                "import os,sys,qdrant_client,streamlit,rank_bm25; "
                "print(sys.executable); print(os.getcwd()); print('deps=OK')"
            ),
        ],
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    check("Child worker uses same Python/dependencies", child.returncode == 0, child.stdout.strip())
    child_lines = [line.strip() for line in (child.stdout or "").splitlines() if line.strip()]
    if child_lines:
        check("Child worker project cwd is pinned", str(ROOT.resolve()).casefold() in " ".join(child_lines).casefold(), child.stdout.strip())

    try:
        plan = get_update_plan()
        check("Read-only update plan executes", isinstance(plan, dict) and plan.get("mode") in {"noop", "full_rebuild"}, {"mode": plan.get("mode"), "reason": plan.get("reason")})
    except Exception as exc:
        plan = {"documents": []}
        check("Read-only update plan executes", False, f"{type(exc).__name__}: {exc}")

    try:
        preflight = get_kb_update_preflight(plan.get("documents") or [])
        check("Cross-PC KB preflight passes", bool(preflight.get("ok")), preflight.get("issues") or preflight.get("notes"))
        write_probes = ((preflight.get("filesystem") or {}).get("write_probes") or {})
        check("Storage/log write probes pass", bool(write_probes) and all(bool(v.get("writable")) for v in write_probes.values()), write_probes)
        disk = ((preflight.get("filesystem") or {}).get("disk") or {})
        check("Transactional disk-space measurement available", bool(disk.get("free_bytes")) and bool(disk.get("required_free_bytes")), disk)
    except Exception as exc:
        check("Cross-PC KB preflight passes", False, f"{type(exc).__name__}: {exc}")

    try:
        with _update_lock("portability_test"):
            lock_exists_inside = True
        check("Cross-process update lock can be acquired/released", lock_exists_inside, "lock probe only; no KB write")
    except Exception as exc:
        check("Cross-process update lock can be acquired/released", False, f"{type(exc).__name__}: {exc}")

    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    check("Streamlit button routes through isolated worker", "launch_kb_update_subprocess" in app_source and "smart_build()" not in app_source)
    check("LAN web-update guard remains active", "LAN_SERVER_MODE" in app_source and "ALLOW_WEB_KB_UPDATE" in app_source)

    failed = [item for item in checks if not item["passed"]]
    result = {
        "version": "v6.5.3",
        "test": "KB Update Portability Dry Run",
        "created": datetime.now().isoformat(timespec="seconds"),
        "overall": "PASS" if not failed else "FAIL",
        "failed_checks": len(failed),
        "check_count": len(checks),
        "no_kb_rebuild_performed": True,
        "checks": checks,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("version", "test", "overall", "failed_checks", "check_count", "no_kb_rebuild_performed")}, indent=2))
    print("Output:", OUT)
    if failed:
        for item in failed:
            print("[FAIL]", item["name"], "::", item.get("detail", ""))
        return 1
    print("[PASS] KB Update portability dry run completed without rebuilding the KB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
