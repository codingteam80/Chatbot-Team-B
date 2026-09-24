from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from config.settings import TECHNICAL_DOCUMENT_DIR
from scripts.kb_health import run_health_check
from scripts.kb_update_runner import run_kb_update_worker
from scripts.smart_build import get_update_plan

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "logs" / "kb_lifecycle_probe"
PROBE_NAME = "__docubot_v655_lifecycle_probe__.txt"
PROBE_PATH = Path(TECHNICAL_DOCUMENT_DIR) / PROBE_NAME


def _plan_snapshot() -> dict:
    plan = get_update_plan()
    changes = plan.get("changes") or {}
    return {
        "mode": plan.get("mode"),
        "reason": plan.get("reason"),
        "added": list(changes.get("added") or []),
        "updated": list(changes.get("updated") or []),
        "deleted": list(changes.get("deleted") or []),
        "unchanged": list(changes.get("unchanged") or []),
    }


def _contains(values, name: str) -> bool:
    target = name.casefold()
    return any(str(value).casefold() == target for value in values or [])


def _run_stage(label: str, expected_change: str) -> dict:
    before = _plan_snapshot()
    change_values = before.get(expected_change) or []
    plan_ok = before.get("mode") == "incremental" and _contains(change_values, PROBE_NAME)
    result = run_kb_update_worker(source=f"v6.5.5_lifecycle_{label}") if plan_ok else {
        "overall": "BLOCKED",
        "stage": "plan_validation",
        "message": f"Expected incremental/{expected_change} plan for {PROBE_NAME}",
    }
    after = _plan_snapshot()
    details = result.get("build_details") or {}
    stage_ok = bool(
        plan_ok
        and result.get("overall") == "PASS"
        and result.get("build_action") == "transactional_incremental"
        and after.get("mode") == "noop"
    )
    if expected_change in {"added", "updated"}:
        stage_ok = stage_ok and int(details.get("embedded_changed_chunks", 0) or 0) >= 1
    if expected_change in {"updated", "deleted"}:
        stage_ok = stage_ok and int(details.get("removed_old_chunks", 0) or 0) >= 1
    if expected_change == "deleted":
        stage_ok = stage_ok and int(details.get("embedded_changed_chunks", 0) or 0) == 0

    return {
        "label": label,
        "expected_change": expected_change,
        "plan_before": before,
        "update_result": result,
        "plan_after": after,
        "pass": bool(stage_ok),
    }


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    initial = _plan_snapshot()
    summary = {
        "version": "v6.5.5",
        "created": datetime.now().isoformat(timespec="seconds"),
        "probe_file": str(PROBE_PATH),
        "initial_plan": initial,
        "stages": [],
        "recovery": None,
        "overall": "FAIL",
    }

    if initial.get("mode") != "noop":
        summary["message"] = (
            "Live lifecycle probe requires a clean noop baseline. Apply pending user document changes first."
        )
    elif PROBE_PATH.exists():
        summary["message"] = f"Probe path already exists: {PROBE_PATH}"
    else:
        try:
            PROBE_PATH.write_text(
                "DocuBot lifecycle probe v1. This temporary file validates incremental add.\n",
                encoding="utf-8",
            )
            summary["stages"].append(_run_stage("add", "added"))
            if not summary["stages"][-1]["pass"]:
                raise RuntimeError("add stage failed")

            PROBE_PATH.write_text(
                "DocuBot lifecycle probe v2. This changed content validates incremental modify.\n",
                encoding="utf-8",
            )
            summary["stages"].append(_run_stage("modify", "updated"))
            if not summary["stages"][-1]["pass"]:
                raise RuntimeError("modify stage failed")

            PROBE_PATH.unlink()
            summary["stages"].append(_run_stage("delete", "deleted"))
            if not summary["stages"][-1]["pass"]:
                raise RuntimeError("delete stage failed")

            final_plan = _plan_snapshot()
            health_rc = int(run_health_check())
            summary["final_plan"] = final_plan
            summary["kb_health_exit_code"] = health_rc
            summary["overall"] = (
                "PASS" if final_plan.get("mode") == "noop" and health_rc == 0 else "FAIL"
            )
            summary["message"] = (
                "Incremental add/modify/delete lifecycle returned to the original clean KB state."
                if summary["overall"] == "PASS"
                else "Lifecycle stages ran, but final noop/KB Health verification did not pass."
            )
        except Exception as error:
            summary["message"] = f"{type(error).__name__}: {error}"
        finally:
            # Fail-safe cleanup: never leave the temporary probe as an intended
            # source document. If a failed stage left it tracked or pending,
            # remove the file and run one guarded recovery update.
            try:
                if PROBE_PATH.exists():
                    PROBE_PATH.unlink()
                recovery_plan = _plan_snapshot()
                if recovery_plan.get("mode") != "noop":
                    recovery = run_kb_update_worker(source="v6.5.5_lifecycle_recovery")
                    summary["recovery"] = recovery
                else:
                    summary["recovery"] = {"overall": "NOT_NEEDED", "plan": recovery_plan}
            except Exception as recovery_error:
                summary["recovery"] = {
                    "overall": "FAIL",
                    "message": f"{type(recovery_error).__name__}: {recovery_error}",
                }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / f"incremental_lifecycle_{stamp}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Overall: {summary['overall']}")
    print(summary.get("message", ""))
    for stage in summary.get("stages") or []:
        details = (stage.get("update_result") or {}).get("build_details") or {}
        print(
            stage.get("label"),
            "PASS" if stage.get("pass") else "FAIL",
            f"embedded={details.get('embedded_changed_chunks')}",
            f"removed={details.get('removed_old_chunks')}",
            f"skipped={details.get('unchanged_skipped')}",
        )
    print(out)
    return 0 if summary.get("overall") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
