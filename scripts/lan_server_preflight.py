from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from config.settings import EMBED_MODEL_NAME, LAN_SERVER_PORT, OLLAMA_COMPLEX_MODEL, OLLAMA_FAST_MODEL
from scripts.lan_common import (
    build_employee_url,
    detect_lan_ipv4_addresses,
    missing_required_models,
    ollama_models,
    port_is_available,
    write_employee_shortcut,
)

ROOT = Path(__file__).resolve().parent.parent
REPORT_ROOT = ROOT / "logs" / "lan_server_validation"


def _run_module(module_name: str) -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", module_name],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return int(result.returncode), result.stdout or ""


def run_preflight(*, require_port_free: bool = True, save_report: bool = True) -> dict:
    created = datetime.now()
    port = int(LAN_SERVER_PORT)
    rows: list[dict] = []

    lock_rc, lock_output = _run_module("scripts.validate_production_architecture_lock")
    rows.append({
        "check": "production_architecture_lock",
        "status": "PASS" if lock_rc == 0 else "FAIL",
        "detail": "Certified production architecture matches the lock." if lock_rc == 0 else "Architecture lock check failed.",
    })

    kb_rc, kb_output = _run_module("scripts.kb_health")
    rows.append({
        "check": "knowledge_base_health",
        "status": "PASS" if kb_rc == 0 else "FAIL",
        "detail": "Knowledge base is healthy." if kb_rc == 0 else "Knowledge base health check failed.",
    })

    ollama_ok, installed, ollama_error = ollama_models()
    missing = missing_required_models(
        installed,
        (OLLAMA_FAST_MODEL, OLLAMA_COMPLEX_MODEL, EMBED_MODEL_NAME),
    )
    model_ok = ollama_ok and not missing
    if model_ok:
        model_detail = (
            "Ollama reachable; required generation/embedding models present: "
            f"{OLLAMA_FAST_MODEL}, {OLLAMA_COMPLEX_MODEL}, {EMBED_MODEL_NAME}."
        )
    elif not ollama_ok:
        model_detail = f"Ollama is not reachable: {ollama_error}"
    else:
        model_detail = "Missing required Ollama model(s): " + ", ".join(missing)
    rows.append({
        "check": "ollama_and_models",
        "status": "PASS" if model_ok else "FAIL",
        "detail": model_detail,
    })

    addresses = detect_lan_ipv4_addresses()
    rows.append({
        "check": "lan_ipv4",
        "status": "PASS" if addresses else "FAIL",
        "detail": ", ".join(addresses) if addresses else "No private LAN IPv4 address detected.",
    })

    port_free = port_is_available(port)
    if require_port_free:
        port_status = port_free
        port_detail = (
            f"TCP {port} is available for DocuBot."
            if port_free
            else f"TCP {port} is already in use. Stop the existing service or choose another port."
        )
    else:
        port_status = True
        port_detail = f"TCP {port} availability check skipped for an already-running server."
    rows.append({
        "check": "streamlit_port",
        "status": "PASS" if port_status else "FAIL",
        "detail": port_detail,
    })

    overall = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    urls = [build_employee_url(address, port) for address in addresses]

    shortcut = ""
    if overall == "PASS" and urls:
        shortcut = str(write_employee_shortcut(ROOT, urls[0]).resolve())

    result = {
        "created": created.isoformat(timespec="seconds"),
        "release": "v6.4.82-finalized-production",
        "port": port,
        "employee_urls": urls,
        "shortcut": shortcut,
        "checks": rows,
        "overall": overall,
    }

    lines = [
        "DocuBot Centralized LAN Server Preflight",
        "=" * 76,
        f"Created       : {result['created']}",
        f"Release       : {result['release']}",
        f"Overall       : {overall}",
        "",
        "Checks",
        "-" * 76,
    ]
    for row in rows:
        lines.append(f"[{row['status']:<4}] {row['check']}: {row['detail']}")
    lines.extend(["", "Employee access URL(s)", "-" * 76])
    if urls:
        lines.extend(urls)
    else:
        lines.append("No LAN URL available.")
    if shortcut:
        lines.extend(["", f"Shortcut written: {shortcut}"])
    lines.extend([
        "",
        "Security note",
        "-" * 76,
        "Expose TCP 8501 only to the trusted office LocalSubnet/Private network profile.",
        "Do not port-forward DocuBot to the public Internet.",
    ])
    summary = "\n".join(lines) + "\n"

    if save_report:
        run_dir = REPORT_ROOT / ("run_" + created.strftime("%Y%m%d_%H%M%S"))
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.txt").write_text(summary, encoding="utf-8")
        (run_dir / "results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "architecture_lock.txt").write_text(lock_output, encoding="utf-8")
        (run_dir / "kb_health.txt").write_text(kb_output, encoding="utf-8")
        result["report_dir"] = str(run_dir.resolve())

    print(summary, end="")
    if result.get("report_dir"):
        print(f"Report folder : {result['report_dir']}")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate DocuBot centralized LAN server readiness")
    parser.add_argument("--allow-port-in-use", action="store_true")
    args = parser.parse_args(argv)
    result = run_preflight(require_port_free=not args.allow_port_in_use)
    return 0 if result["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
