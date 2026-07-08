import threading
import subprocess
import sys

from pdf_server import start_pdf_server


threading.Thread(
    target=start_pdf_server,
    daemon=True
).start()


subprocess.run(
    [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py"
    ]
)