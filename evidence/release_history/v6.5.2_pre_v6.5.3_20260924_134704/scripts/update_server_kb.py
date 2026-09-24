from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from config.settings import LAN_SERVER_PORT
from scripts.kb_health import run_health_check
from scripts.lan_common import local_tcp_port_is_open
from scripts.smart_build import get_update_plan, smart_build

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

    plan = get_update_plan()
    print("DocuBot Server Knowledge Base Update")
    print("=" * 72)
    print("Plan mode :", plan.get("mode"))
    if plan.get("reason"):
        print("Reason    :", plan.get("reason"))

    if plan.get("mode") == "noop":
        print("No source-document changes detected. Nothing to update.")
        return run_health_check()

    ok = bool(smart_build())
    health_rc = run_health_check()

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "plan": _json_safe(plan),
        "build_ok": ok,
        "kb_health_exit_code": int(health_rc),
        "overall": "PASS" if ok and health_rc == 0 else "FAIL",
    }
    (REPORT_ROOT / f"server_kb_update_{stamp}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Overall   :", report["overall"])
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
