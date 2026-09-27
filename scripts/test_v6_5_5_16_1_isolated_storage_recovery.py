from __future__ import annotations

# IMPORTANT: set the isolated storage override before importing any DocuBot
# modules because config.settings resolves storage paths at import time.
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR = ROOT / "logs" / "v6_5_5_16_1_storage_recovery_probe"
OUTDIR.mkdir(parents=True, exist_ok=True)
ISOLATED_STORAGE = OUTDIR / f"isolated_storage_{STAMP}"
os.environ["DOCUBOT_OPTION_C_STORAGE_DIR"] = str(ISOLATED_STORAGE)
os.environ.setdefault("PYTHONUTF8", "1")

from config import settings
from scripts import build_qdrant_index, smart_build
from retrieval.qdrant_search import qdrant_collection_count
from retrieval.bm25_index import CORPUS_FILE, INDEX_FILE
from utils import file_utils
from utils.manifest import ManifestManager

RESULT_JSON = OUTDIR / f"storage_recovery_probe_{STAMP}.json"
RESULT_ZIP = OUTDIR / f"DocuBot_v6.5.5.16.1_Storage_Recovery_Probe_Result_{STAMP}.zip"
RESULT_SHA = RESULT_ZIP.with_suffix(RESULT_ZIP.suffix + ".sha256.txt")
PROBE_DOCUMENT = ROOT / "data" / "technical_documents" / "Employee_Leave_Test.txt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return value


def main() -> int:
    started = time.perf_counter()
    result = {
        "version": "v6.5.5.16.1",
        "test": "isolated unreadable-storage -> source embedding recovery",
        "created": datetime.now().isoformat(timespec="seconds"),
        "production_storage_untouched_by_design": True,
        "production_storage": str(settings.OPTION_C_V4_PRODUCTION_STORAGE_DIR),
        "isolated_storage": str(ISOLATED_STORAGE),
        "probe_document": str(PROBE_DOCUMENT),
        "overall": "FAIL",
        "checks": {},
    }

    # Keep this proof fast and safe: use one existing small office-policy source
    # while exercising the exact full-rebuild/Qwen3/Qdrant transaction code.
    if not PROBE_DOCUMENT.is_file():
        result["error"] = f"Probe document missing: {PROBE_DOCUMENT}"
        return finish(result, started)

    only_probe = lambda: [PROBE_DOCUMENT]
    smart_build.get_all_documents = only_probe
    build_qdrant_index.get_all_documents = only_probe
    file_utils.get_all_documents = only_probe

    try:
        shutil.rmtree(ISOLATED_STORAGE, ignore_errors=True)
        settings.QDRANT_DIR.parent.mkdir(parents=True, exist_ok=True)

        # Deliberately make the configured Qdrant path unreadable as a local
        # vector database. This reproduces the class of deployment condition
        # that previously required manually copying a healthy storage folder.
        settings.QDRANT_DIR.write_text(
            "INTENTIONALLY_UNREADABLE_QDRANT_STORAGE\n",
            encoding="utf-8",
        )

        plan_before = smart_build.get_update_plan()
        result["plan_before"] = json_safe(plan_before)
        result["checks"]["planned_full_rebuild"] = plan_before.get("mode") == "full_rebuild"
        result["checks"]["recovery_required_detected"] = bool(plan_before.get("qdrant_recovery_required"))
        result["checks"]["unreadable_error_captured"] = bool(plan_before.get("collection_error"))

        preflight = smart_build.get_kb_update_preflight(
            plan_before.get("documents") or [],
            requires_embedding=True,
            allow_unreadable_qdrant_recovery=True,
        )
        result["preflight"] = json_safe(preflight)
        result["checks"]["recovery_preflight_passed"] = bool(preflight.get("ok"))
        result["checks"]["recovery_preflight_used"] = bool((preflight.get("qdrant_recovery") or {}).get("used"))

        build_ok = bool(smart_build.smart_build()) if preflight.get("ok") else False
        build_details = smart_build.get_last_build_result()
        result["build_ok"] = build_ok
        result["build_details"] = json_safe(build_details)
        result["checks"]["source_rebuild_completed"] = build_ok
        result["checks"]["recovery_recorded_by_build"] = bool(build_details.get("qdrant_recovery_used"))

        plan_after = smart_build.get_update_plan() if build_ok else {}
        result["plan_after"] = json_safe(plan_after)
        result["checks"]["post_plan_noop"] = plan_after.get("mode") == "noop"

        qdrant_count = qdrant_collection_count() if build_ok else 0
        manifest = ManifestManager.load() if build_ok else {}
        result["qdrant_count"] = int(qdrant_count or 0)
        result["manifest_entries"] = len(manifest)
        result["bm25_files"] = {
            "corpus": str(CORPUS_FILE),
            "corpus_exists": CORPUS_FILE.is_file(),
            "index": str(INDEX_FILE),
            "index_exists": INDEX_FILE.is_file(),
        }
        result["checks"]["qdrant_has_embedded_points"] = int(qdrant_count or 0) >= 1
        result["checks"]["manifest_tracks_probe"] = len(manifest) == 1
        result["checks"]["bm25_built"] = CORPUS_FILE.is_file() and INDEX_FILE.is_file()
        result["checks"]["active_qdrant_replaced_with_directory"] = settings.QDRANT_DIR.is_dir()
        result["checks"]["no_transaction_backup_left"] = not any(
            settings.QDRANT_DIR.parent.glob(settings.QDRANT_DIR.name + "__backup__*")
        )
        result["checks"]["no_transaction_staging_left"] = not any(
            settings.QDRANT_DIR.parent.glob(settings.QDRANT_DIR.name + "__staging__*")
        )

        result["overall"] = "PASS" if all(result["checks"].values()) else "FAIL"
    except Exception as error:
        import traceback
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    finally:
        # The probe is intentionally isolated. Remove its temporary vector/BM25
        # data after recording the proof so production storage is never touched.
        try:
            shutil.rmtree(ISOLATED_STORAGE, ignore_errors=True)
            result["isolated_storage_cleaned"] = not ISOLATED_STORAGE.exists()
        except Exception as cleanup_error:
            result["isolated_storage_cleaned"] = False
            result["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"

    return finish(result, started)


def finish(result: dict, started: float) -> int:
    result["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    RESULT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with zipfile.ZipFile(RESULT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(RESULT_JSON, arcname=RESULT_JSON.name)
    digest = sha256(RESULT_ZIP)
    RESULT_SHA.write_text(f"{digest}  {RESULT_ZIP.name}\n", encoding="utf-8")
    print(json.dumps({
        "version": result.get("version"),
        "test": result.get("test"),
        "overall": result.get("overall"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "qdrant_count": result.get("qdrant_count"),
        "checks": result.get("checks"),
    }, indent=2, ensure_ascii=False))
    print("Result ZIP:", RESULT_ZIP)
    print("SHA256    :", RESULT_SHA)
    return 0 if result.get("overall") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
