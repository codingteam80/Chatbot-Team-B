from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from config.settings import LAN_SERVER_PORT
from scripts.kb_update_runner import run_kb_update_worker
from scripts.lan_common import local_tcp_port_is_open

ROOT = Path(__file__).resolve().parent.parent
REPORT_ROOT = ROOT / "logs" / "server_kb_update"


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def main() -> int:
    if local_tcp_port_is_open(int(LAN_SERVER_PORT)):
        print("[FAIL] DocuBot appears to be running on TCP", LAN_SERVER_PORT)
        print("Stop Start_DocuBot_LAN_Server.bat first, then run this update again.")
        print("This guard prevents employee queries from reading the index while it is being updated.")
        return 2

    print("DocuBot Server Knowledge Base Update")
    print("=" * 72)
    report = run_kb_update_worker(source="server_bat")

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    server_report = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "worker": _json_safe(report),
        "overall": report.get("overall", "FAIL"),
    }
    path = REPORT_ROOT / f"server_kb_update_{stamp}.json"
    path.write_text(
        json.dumps(server_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Overall   :", report.get("overall"))
    print("Stage     :", report.get("stage"))
    print("Message   :", report.get("message"))
    print("Report    :", report.get("report_path", path))
    return 0 if report.get("overall") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
