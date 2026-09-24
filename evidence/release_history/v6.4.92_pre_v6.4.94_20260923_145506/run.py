from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

from pdf_server import start_pdf_server

ROOT = Path(__file__).resolve().parent


def main() -> int:
    """Normal DocuBot entrypoint used by VS Code and local production runs."""
    os.chdir(ROOT)

    threading.Thread(
        target=start_pdf_server,
        daemon=True,
    ).start()

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "app.py"),
        ],
        cwd=ROOT,
        env=env,
        check=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
