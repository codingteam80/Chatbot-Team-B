from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    """Normal DocuBot entrypoint used by VS Code and local production runs.

    Source files are opened directly from the canonical technical_documents
    folder (or downloaded through Streamlit in LAN mode).  No auxiliary source
    HTTP server is started by the production launcher.
    """

    os.chdir(ROOT)

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
