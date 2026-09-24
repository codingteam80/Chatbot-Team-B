from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from config.settings import LAN_SERVER_PORT
from scripts.lan_server_preflight import run_preflight

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    os.chdir(ROOT)

    print("=" * 76)
    print("DocuBot v6.4.82 - FINALIZED PRODUCTION LAN SERVER")
    print("=" * 76)
    print("Running preflight checks before exposing DocuBot to the office LAN...\n")

    preflight = run_preflight(require_port_free=True, save_report=True)
    if preflight.get("overall") != "PASS":
        print("\nLAN server was NOT started because preflight failed.")
        return 1

    urls = list(preflight.get("employee_urls") or [])
    print("\nDocuBot will listen on all LAN interfaces.")
    for url in urls:
        print(f"Employee URL: {url}")
    print("Employees need only a browser; do not copy server-side Qdrant/storage data to client PCs.")
    print("Press Ctrl+C in this window to stop the central server.\n")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["DOCUBOT_LAN_SERVER_MODE"] = "1"
    env["DOCUBOT_ALLOW_WEB_KB_UPDATE"] = "0"
    env["DOCUBOT_LAN_PORT"] = str(int(LAN_SERVER_PORT))
    # Production LAN launch must never inherit the QA-only forced-model switch.
    env.pop("DOCUBOT_OLLAMA_MODEL", None)

    command = [
        sys.executable,
        "-X",
        "utf8",
        "-u",
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.address=0.0.0.0",
        f"--server.port={int(LAN_SERVER_PORT)}",
        "--server.headless=true",
        "--server.enableXsrfProtection=true",
        "--browser.gatherUsageStats=false",
        "--server.fileWatcherType=none",
        "--server.runOnSave=false",
    ]

    try:
        completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
        return int(completed.returncode)
    except KeyboardInterrupt:
        print("\nDocuBot LAN server stopped by administrator.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
