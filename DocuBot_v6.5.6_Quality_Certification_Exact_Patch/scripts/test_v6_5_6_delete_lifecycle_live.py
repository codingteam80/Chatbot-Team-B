from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import (
    BM25_DIR,
    METADATA_DIR,
    QDRANT_COLLECTION_NAME,
    QDRANT_DIR,
    TECHNICAL_DOCUMENT_DIR,
)
from retrieval.qdrant_search import _qdrant_imports
from scripts.kb_health import run_health_check
from scripts.kb_update_runner import run_kb_update_worker
from scripts.smart_build import get_update_plan
from utils.manifest import ManifestManager

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "logs" / "v6_5_6_quality_certification"
PROBE_NAME = "__docubot_v656_delete_lifecycle_probe__.txt"
PROBE_PATH = Path(TECHNICAL_DOCUMENT_DIR) / PROBE_NAME
PROBE_TOKEN = "docubotv656deletelifecycleprobe"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _plan_snapshot() -> dict[str, Any]:
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


def _bm25_records() -> list[dict[str, Any]]:
    path = Path(BM25_DIR) / "corpus.pkl"
    if not path.is_file():
        return []
    with path.open("rb") as handle:
        value = pickle.load(handle)
    return list(value or []) if isinstance(value, list) else []


def _bm25_probe_count(records: list[dict[str, Any]]) -> int:
    target = PROBE_NAME.casefold()
    return sum(
        1
        for item in records
        if str(((item.get("metadata") or {}).get("file_name") or "")).casefold() == target
        or PROBE_TOKEN in str(item.get("text") or "").casefold()
    )


def _bm25_file_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in records:
        name = str(((item.get("metadata") or {}).get("file_name") or "")).casefold()
        if name:
            counts[name] = counts.get(name, 0) + 1
    return counts


def _qdrant_file_counts() -> tuple[int, dict[str, int]]:
    QdrantClient = _qdrant_imports()
    client = QdrantClient(path=str(QDRANT_DIR))
    try:
        counts: dict[str, int] = {}
        total = 0
        offset = None
        while True:
            response = client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if isinstance(response, tuple):
                points, next_offset = response
            else:
                points = getattr(response, "points", None) or []
                next_offset = getattr(response, "next_page_offset", None)
            for point in points or []:
                payload = dict(getattr(point, "payload", None) or {})
                metadata = payload.get("metadata") or {}
                name = str((metadata.get("file_name") or "")).casefold()
                if name:
                    counts[name] = counts.get(name, 0) + 1
                total += 1
            if next_offset is None:
                break
            offset = next_offset
        return total, counts
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def _semantic_index_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for path in sorted(Path(METADATA_DIR).glob("misra_semantic_rule_index*.json")):
        if path.is_file():
            output[path.name] = _sha256(path)
    return output


def _manifest_snapshot() -> dict[str, dict[str, Any]]:
    manifest = ManifestManager.load()
    return {
        key: {
            "file_name": str(info.get("file_name") or ""),
            "hash": str(info.get("hash") or ""),
            "last_indexed": str(info.get("last_indexed") or ""),
            "index_status": str(info.get("index_status") or ""),
        }
        for key, info in manifest.items()
    }


def _state_snapshot() -> dict[str, Any]:
    bm25 = _bm25_records()
    qtotal, qcounts = _qdrant_file_counts()
    manifest = _manifest_snapshot()
    key = ManifestManager.document_key(PROBE_NAME)
    return {
        "manifest": manifest,
        "manifest_probe_present": key in manifest,
        "bm25_total": len(bm25),
        "bm25_file_counts": _bm25_file_counts(bm25),
        "bm25_probe_count": _bm25_probe_count(bm25),
        "qdrant_total": qtotal,
        "qdrant_file_counts": qcounts,
        "qdrant_probe_count": int(qcounts.get(PROBE_NAME.casefold(), 0)),
        "semantic_index_hashes": _semantic_index_hashes(),
    }


def _baseline_preserved(before: dict[str, Any], after: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    before_manifest = before.get("manifest") or {}
    after_manifest = after.get("manifest") or {}
    for key, info in before_manifest.items():
        current = after_manifest.get(key)
        if current is None:
            reasons.append(f"baseline manifest entry disappeared: {key}")
            continue
        for field in ("hash", "last_indexed", "index_status"):
            if str(current.get(field) or "") != str(info.get(field) or ""):
                reasons.append(f"baseline manifest field changed: {key}.{field}")
    if before.get("semantic_index_hashes") != after.get("semantic_index_hashes"):
        reasons.append("MISRA semantic Rule index hash changed during unrelated probe lifecycle")
    for file_name, count in (before.get("qdrant_file_counts") or {}).items():
        if int((after.get("qdrant_file_counts") or {}).get(file_name, -1)) != int(count):
            reasons.append(f"baseline Qdrant chunk count changed: {file_name}")
    for file_name, count in (before.get("bm25_file_counts") or {}).items():
        if int((after.get("bm25_file_counts") or {}).get(file_name, -1)) != int(count):
            reasons.append(f"baseline BM25 chunk count changed: {file_name}")
    if int(before.get("qdrant_total") or 0) != int(after.get("qdrant_total") or 0):
        reasons.append("final Qdrant total did not return to baseline")
    if int(before.get("bm25_total") or 0) != int(after.get("bm25_total") or 0):
        reasons.append("final BM25 total did not return to baseline")
    return (not reasons), reasons


def _run_stage(label: str, expected_change: str) -> dict[str, Any]:
    before_plan = _plan_snapshot()
    change_values = before_plan.get(expected_change) or []
    plan_ok = before_plan.get("mode") == "incremental" and _contains(change_values, PROBE_NAME)
    result = run_kb_update_worker(source=f"v6.5.6_delete_lifecycle_{label}") if plan_ok else {
        "overall": "BLOCKED",
        "stage": "plan_validation",
        "message": f"Expected incremental/{expected_change} plan for {PROBE_NAME}",
    }
    after_plan = _plan_snapshot()
    state = _state_snapshot()
    details = result.get("build_details") or {}

    presence_expected = expected_change != "deleted"
    state_ok = (
        bool(state.get("manifest_probe_present")) is presence_expected
        and ((int(state.get("bm25_probe_count") or 0) > 0) is presence_expected)
        and ((int(state.get("qdrant_probe_count") or 0) > 0) is presence_expected)
    )
    stage_ok = bool(
        plan_ok
        and result.get("overall") == "PASS"
        and result.get("build_action") == "transactional_incremental"
        and after_plan.get("mode") == "noop"
        and state_ok
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
        "plan_before": before_plan,
        "update_result": result,
        "plan_after": after_plan,
        "state_after": state,
        "state_presence_expected": presence_expected,
        "state_presence_verified": state_ok,
        "pass": bool(stage_ok),
    }


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    initial_plan = _plan_snapshot()
    summary: dict[str, Any] = {
        "version": "v6.5.6",
        "created": datetime.now().isoformat(timespec="seconds"),
        "probe_file": str(PROBE_PATH),
        "probe_token": PROBE_TOKEN,
        "initial_plan": initial_plan,
        "stages": [],
        "recovery": None,
        "overall": "FAIL",
    }

    if initial_plan.get("mode") != "noop":
        summary["message"] = "Delete lifecycle requires a clean noop baseline. Apply pending user document changes first."
    elif PROBE_PATH.exists():
        summary["message"] = f"Probe path already exists: {PROBE_PATH}"
    else:
        baseline: dict[str, Any] | None = None
        try:
            baseline = _state_snapshot()
            summary["baseline_state"] = baseline

            PROBE_PATH.write_text(
                f"{PROBE_TOKEN} v1. Temporary DocuBot probe validates transactional incremental add.\n",
                encoding="utf-8",
            )
            summary["stages"].append(_run_stage("add", "added"))
            if not summary["stages"][-1]["pass"]:
                raise RuntimeError("add stage failed")

            PROBE_PATH.write_text(
                f"{PROBE_TOKEN} v2. Changed temporary DocuBot probe validates transactional incremental modify.\n",
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
            final_state = _state_snapshot()
            preserved, preservation_reasons = _baseline_preserved(baseline, final_state)
            health_rc = int(run_health_check())
            summary["final_plan"] = final_plan
            summary["final_state"] = final_state
            summary["baseline_preserved"] = preserved
            summary["baseline_preservation_reasons"] = preservation_reasons
            summary["kb_health_exit_code"] = health_rc
            summary["overall"] = "PASS" if (
                final_plan.get("mode") == "noop"
                and health_rc == 0
                and preserved
                and not final_state.get("manifest_probe_present")
                and int(final_state.get("bm25_probe_count") or 0) == 0
                and int(final_state.get("qdrant_probe_count") or 0) == 0
            ) else "FAIL"
            summary["message"] = (
                "Actual add/modify/delete lifecycle passed; deleted probe is absent from manifest/Qdrant/BM25 and unchanged baseline content was preserved."
                if summary["overall"] == "PASS"
                else "Lifecycle ran but final delete/baseline-preservation/KB-Health verification failed."
            )
        except Exception as error:
            summary["message"] = f"{type(error).__name__}: {error}"
        finally:
            try:
                if PROBE_PATH.exists():
                    PROBE_PATH.unlink()
                recovery_plan = _plan_snapshot()
                if recovery_plan.get("mode") != "noop":
                    summary["recovery"] = run_kb_update_worker(source="v6.5.6_delete_lifecycle_recovery")
                else:
                    summary["recovery"] = {"overall": "NOT_NEEDED", "plan": recovery_plan}
            except Exception as recovery_error:
                summary["recovery"] = {"overall": "FAIL", "message": f"{type(recovery_error).__name__}: {recovery_error}"}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / f"v6.5.6_delete_lifecycle_{stamp}.json"
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
            f"manifest/bm25/qdrant_presence={stage.get('state_presence_verified')}",
        )
    print(out)
    return 0 if summary.get("overall") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
