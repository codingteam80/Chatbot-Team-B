from __future__ import annotations

import hashlib
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    BM25_DIR,
    METADATA_DIR,
    QDRANT_COLLECTION_NAME,
    QDRANT_DIR,
    TECHNICAL_DOCUMENT_DIR,
)
from scripts.kb_health import run_health_check
from scripts.kb_update_runner import run_kb_update_worker
from scripts.smart_build import get_update_plan

OUT_ROOT = ROOT / "logs" / "kb_lifecycle_probe"
PROBE_NAME = "__docubot_v655_lifecycle_probe__.txt"
PROBE_PATH = Path(TECHNICAL_DOCUMENT_DIR) / PROBE_NAME
MANIFEST_PATH = Path(METADATA_DIR) / "manifest.json"
BM25_CORPUS_PATH = Path(BM25_DIR) / "corpus.pkl"
SEMANTIC_INDEX_FILES = (
    Path(METADATA_DIR) / "misra_semantic_rule_index_v1.json",
    Path(METADATA_DIR) / "misra_semantic_rule_index_v1.npz",
)


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


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        return {}
    try:
        value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_index_hashes() -> dict[str, str]:
    return {path.name: _sha256(path) for path in SEMANTIC_INDEX_FILES}


def _record_is_probe(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    metadata = record.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    candidates = [
        metadata.get("file_name"),
        metadata.get("source_relative_path"),
        metadata.get("file_path"),
        metadata.get("indexed_file_path"),
    ]
    needle = PROBE_NAME.casefold()
    return any(needle in str(value or "").casefold() for value in candidates)


def _bm25_snapshot() -> dict[str, Any]:
    if not BM25_CORPUS_PATH.is_file():
        return {"readable": False, "total_records": None, "probe_records": None, "error": "corpus.pkl missing"}
    try:
        with BM25_CORPUS_PATH.open("rb") as handle:
            records = pickle.load(handle)
        records = list(records or [])
        probe = sum(1 for record in records if _record_is_probe(record))
        return {"readable": True, "total_records": len(records), "probe_records": probe, "error": ""}
    except Exception as error:
        return {
            "readable": False,
            "total_records": None,
            "probe_records": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _qdrant_snapshot() -> dict[str, Any]:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(path=str(QDRANT_DIR))
        try:
            if hasattr(client, "collection_exists") and not client.collection_exists(QDRANT_COLLECTION_NAME):
                return {"readable": False, "total_points": None, "probe_points": None, "error": "collection missing"}
            total = 0
            probe = 0
            offset = None
            while True:
                points, offset = client.scroll(
                    collection_name=QDRANT_COLLECTION_NAME,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                total += len(points)
                for point in points:
                    payload = getattr(point, "payload", None) or {}
                    if _record_is_probe(payload):
                        probe += 1
                if offset is None:
                    break
            return {"readable": True, "total_points": total, "probe_points": probe, "error": ""}
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
    except Exception as error:
        return {
            "readable": False,
            "total_points": None,
            "probe_points": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _manifest_snapshot() -> dict[str, Any]:
    manifest = _load_manifest()
    needle = PROBE_NAME.casefold()
    probe_entries = []
    for key, info in manifest.items():
        info = info if isinstance(info, dict) else {}
        candidates = [key, info.get("file_name"), info.get("source_relative_path"), info.get("file_path")]
        if any(needle in str(value or "").casefold() for value in candidates):
            probe_entries.append(str(key))
    return {
        "entry_count": len(manifest),
        "probe_entries": probe_entries,
        "full_manifest": manifest,
    }


def _storage_snapshot() -> dict[str, Any]:
    return {
        "manifest": _manifest_snapshot(),
        "bm25": _bm25_snapshot(),
        "qdrant": _qdrant_snapshot(),
        "semantic_rule_index_sha256": _semantic_index_hashes(),
    }


def _probe_presence_ok(snapshot: dict[str, Any], *, expected_present: bool) -> bool:
    manifest_probe = len(((snapshot.get("manifest") or {}).get("probe_entries") or []))
    bm25 = snapshot.get("bm25") or {}
    qdrant = snapshot.get("qdrant") or {}
    if not bm25.get("readable") or not qdrant.get("readable"):
        return False
    bm25_probe = int(bm25.get("probe_records") or 0)
    qdrant_probe = int(qdrant.get("probe_points") or 0)
    if expected_present:
        return manifest_probe >= 1 and bm25_probe >= 1 and qdrant_probe >= 1
    return manifest_probe == 0 and bm25_probe == 0 and qdrant_probe == 0


def _run_stage(label: str, expected_change: str) -> dict[str, Any]:
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
    expected_present = expected_change != "deleted"
    storage_after = _storage_snapshot() if result.get("overall") == "PASS" else {}

    stage_ok = bool(
        plan_ok
        and result.get("overall") == "PASS"
        and result.get("build_action") == "transactional_incremental"
        and after.get("mode") == "noop"
        and _probe_presence_ok(storage_after, expected_present=expected_present)
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
        "storage_after": storage_after,
        "probe_expected_present": expected_present,
        "pass": bool(stage_ok),
    }


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    initial = _plan_snapshot()
    baseline_storage = _storage_snapshot()
    baseline_manifest = (baseline_storage.get("manifest") or {}).get("full_manifest") or {}
    baseline_semantic = baseline_storage.get("semantic_rule_index_sha256") or {}
    baseline_bm25_total = (baseline_storage.get("bm25") or {}).get("total_records")
    baseline_qdrant_total = (baseline_storage.get("qdrant") or {}).get("total_points")

    summary: dict[str, Any] = {
        "version": "v6.5.5.5-quality-check",
        "created": datetime.now().isoformat(timespec="seconds"),
        "probe_file": str(PROBE_PATH),
        "initial_plan": initial,
        "baseline_storage": baseline_storage,
        "stages": [],
        "recovery": None,
        "overall": "FAIL",
    }

    baseline_storage_ok = bool(
        (baseline_storage.get("bm25") or {}).get("readable")
        and (baseline_storage.get("qdrant") or {}).get("readable")
        and _probe_presence_ok(baseline_storage, expected_present=False)
    )

    if initial.get("mode") != "noop":
        summary["message"] = (
            "Live lifecycle probe requires a clean noop baseline. Apply pending user document changes first."
        )
    elif PROBE_PATH.exists():
        summary["message"] = f"Probe path already exists: {PROBE_PATH}"
    elif not baseline_storage_ok:
        summary["message"] = "Baseline manifest/BM25/Qdrant could not be verified before the lifecycle test."
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
            final_storage = _storage_snapshot()
            final_manifest = (final_storage.get("manifest") or {}).get("full_manifest") or {}
            final_bm25_total = (final_storage.get("bm25") or {}).get("total_records")
            final_qdrant_total = (final_storage.get("qdrant") or {}).get("total_points")
            final_semantic = final_storage.get("semantic_rule_index_sha256") or {}

            preservation = {
                "probe_absent_manifest_bm25_qdrant": _probe_presence_ok(final_storage, expected_present=False),
                "manifest_restored_exactly": final_manifest == baseline_manifest,
                "bm25_record_count_restored": final_bm25_total == baseline_bm25_total,
                "qdrant_point_count_restored": final_qdrant_total == baseline_qdrant_total,
                "semantic_rule_index_unchanged": final_semantic == baseline_semantic,
            }
            preservation["all_pass"] = all(preservation.values())

            summary["final_plan"] = final_plan
            summary["kb_health_exit_code"] = health_rc
            summary["final_storage"] = final_storage
            summary["preservation"] = preservation
            summary["overall"] = (
                "PASS"
                if final_plan.get("mode") == "noop" and health_rc == 0 and preservation["all_pass"]
                else "FAIL"
            )
            summary["message"] = (
                "Incremental add/modify/delete lifecycle passed; deleted probe is absent from manifest/Qdrant/BM25 and the original KB state was preserved."
                if summary["overall"] == "PASS"
                else "Lifecycle stages ran, but final deletion/preservation verification did not fully pass."
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
        storage = stage.get("storage_after") or {}
        print(
            stage.get("label"),
            "PASS" if stage.get("pass") else "FAIL",
            f"embedded={details.get('embedded_changed_chunks')}",
            f"removed={details.get('removed_old_chunks')}",
            f"skipped={details.get('unchanged_skipped')}",
            f"probe_manifest={len(((storage.get('manifest') or {}).get('probe_entries') or []))}",
            f"probe_bm25={(storage.get('bm25') or {}).get('probe_records')}",
            f"probe_qdrant={(storage.get('qdrant') or {}).get('probe_points')}",
        )
    if summary.get("preservation"):
        print("Preservation:", json.dumps(summary["preservation"], ensure_ascii=False))
    print(out)
    return 0 if summary.get("overall") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
