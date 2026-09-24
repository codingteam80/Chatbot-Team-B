from __future__ import annotations

import importlib.util
import json
import traceback
from datetime import datetime
from pathlib import Path

from config.settings import (
    DOCUMENT_DIR,
    EMBED_MODEL_NAME,
    EMBED_OLLAMA_URL,
    INDEX_SCHEMA_VERSION,
)
from retrieval.bm25_index import CORPUS_FILE, INDEX_FILE
from utils.file_utils import get_all_documents
from utils.manifest import ManifestManager


ROOT = Path(__file__).resolve().parents[1]
KB_UPDATE_LOG_DIR = ROOT / "logs" / "kb_update"


# Common technical/office files have native DocuBot loaders and do not require
# the optional universal parser just to import the ingestion pipeline.
_NATIVE_PARSER_MODULES = {
    ".pdf": ("fitz",),
    ".docx": ("docx",),
    ".pptx": ("pptx",),
    ".xlsx": ("pandas", "openpyxl"),
    ".txt": (),
    ".csv": (),
    ".json": (),
    ".xml": (),
}


def compare_manifests(old_manifest, new_manifest):
    """Classify source-document changes without touching active indexes."""
    added, updated, deleted, unchanged = [], [], [], []

    for document_key, new_info in new_manifest.items():
        old_info = old_manifest.get(document_key)
        if old_info is None:
            added.append(document_key)
            continue

        changed = (
            new_info.get("hash") != old_info.get("hash")
            or ManifestManager.index_signature(new_info)
            != ManifestManager.index_signature(old_info)
        )
        (updated if changed else unchanged).append(document_key)

    for document_key in old_manifest:
        if document_key not in new_manifest:
            deleted.append(document_key)

    return {
        "added": added,
        "updated": updated,
        "deleted": deleted,
        "unchanged": unchanged,
    }


def bm25_is_ready() -> bool:
    return INDEX_FILE.is_file() and CORPUS_FILE.is_file()


def _manifest_requires_full_rebuild(manifest) -> bool:
    if not manifest:
        return False
    for info in manifest.values():
        if not isinstance(info, dict):
            return True
        if info.get("index_schema_version") != INDEX_SCHEMA_VERSION:
            return True
        if info.get("embedding_model") != EMBED_MODEL_NAME:
            return True
    return False


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _ollama_embedding_preflight() -> tuple[bool, str]:
    """Fast local readiness check before a potentially expensive rebuild."""
    try:
        import requests

        url = str(EMBED_OLLAMA_URL).rstrip("/") + "/api/tags"
        response = requests.get(url, timeout=(2.0, 5.0))
        response.raise_for_status()
        payload = response.json()
    except Exception as error:
        return False, (
            "Ollama is not reachable for KB embeddings. Start Ollama or run "
            "Setup_DocuBot_Production_Environment.bat. "
            f"Detail: {type(error).__name__}: {error}"
        )

    installed = set()
    for item in payload.get("models") or []:
        if not isinstance(item, dict):
            continue
        for key in ("name", "model"):
            value = str(item.get(key) or "").strip()
            if value:
                installed.add(value)

    if EMBED_MODEL_NAME not in installed:
        return False, (
            f"Required embedding model '{EMBED_MODEL_NAME}' is not installed. "
            "Run Setup_DocuBot_Production_Environment.bat."
        )

    return True, ""


def get_kb_update_preflight(documents=None):
    """Return a concise, non-destructive KB-update readiness report."""
    docs = list(documents if documents is not None else get_all_documents())
    issues = []
    notes = []

    if Path(DOCUMENT_DIR).name.casefold() != "technical_documents":
        issues.append(
            "The active knowledge root is not data/technical_documents. "
            "Do not rebuild until config/settings.py is corrected."
        )

    if not docs:
        issues.append(
            "No supported source documents were found under data/technical_documents."
        )

    needed_modules = {"qdrant_client", "rank_bm25", "requests"}
    needs_unstructured = False
    for document in docs:
        ext = Path(document).suffix.lower()
        native_modules = _NATIVE_PARSER_MODULES.get(ext)
        if native_modules is None:
            needs_unstructured = True
        else:
            needed_modules.update(native_modules)

    missing = sorted(name for name in needed_modules if not _module_available(name))
    if missing:
        issues.append(
            "Missing Python dependencies: " + ", ".join(missing)
            + ". Run Setup_DocuBot_Production_Environment.bat."
        )

    if needs_unstructured and not _module_available("unstructured"):
        issues.append(
            "At least one current source file needs the optional 'unstructured' "
            "parser, but it is not installed. Run Setup_DocuBot_Production_Environment.bat."
        )
    elif not needs_unstructured and not _module_available("unstructured"):
        notes.append(
            "unstructured is not installed, but the current source set uses native "
            "DocuBot loaders and can still be rebuilt safely."
        )

    if not issues:
        ollama_ok, ollama_issue = _ollama_embedding_preflight()
        if not ollama_ok:
            issues.append(ollama_issue)

    return {
        "ok": not issues,
        "documents": docs,
        "issues": issues,
        "notes": notes,
        "document_root": str(DOCUMENT_DIR),
    }


def _write_failure_log(stage: str, error=None, details=None) -> Path:
    KB_UPDATE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = KB_UPDATE_LOG_DIR / f"kb_update_failure_{stamp}.txt"
    lines = [
        "DocuBot KB Update Failure",
        "=" * 72,
        f"Stage: {stage}",
        f"Time : {datetime.now().isoformat(timespec='seconds')}",
    ]
    if details:
        lines.append("Details:")
        if isinstance(details, (list, tuple)):
            lines.extend(f"- {item}" for item in details)
        else:
            lines.append(str(details))
    if error is not None:
        lines.append(f"Error: {type(error).__name__}: {error}")
        lines.append("")
        lines.append(traceback.format_exc())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def get_update_plan():
    """Return the canonical production update plan.

    The active source root is data/technical_documents only.  The portable
    manifest format prevents a project copy to a different Windows path from
    being misclassified as a delete+add of every unchanged source file.
    """

    documents = get_all_documents()
    old_manifest = ManifestManager.load()
    new_manifest = ManifestManager.build(documents, previous_manifest=old_manifest)
    changes = compare_manifests(old_manifest, new_manifest)

    try:
        from retrieval.qdrant_search import qdrant_collection_count

        collection_count = qdrant_collection_count()
        collection_error = None
    except Exception as error:
        collection_count = None
        collection_error = str(error)

    reason = None
    has_document_changes = any(
        changes[name] for name in ("added", "updated", "deleted")
    )

    if _manifest_requires_full_rebuild(old_manifest):
        reason = "The production schema/embedding identity changed."
    elif collection_error and documents:
        reason = f"The active Qdrant index could not be opened safely: {collection_error}"
    elif not old_manifest and documents and (collection_count or 0) > 0:
        reason = "An existing Qdrant index has no usable tracking manifest."
    elif old_manifest and documents and collection_count == 0:
        reason = "The active Qdrant index is missing or empty while documents are tracked."
    elif has_document_changes:
        reason = "Source-document changes require the certified transactional Qdrant rebuild."
    elif documents and not bm25_is_ready():
        reason = "BM25 companion data is missing or incomplete."

    return {
        "mode": "full_rebuild" if reason else "noop",
        "reason": reason,
        "documents": documents,
        "old_manifest": old_manifest,
        "new_manifest": new_manifest,
        "changes": changes,
        "collection_count": collection_count,
        "collection_error": collection_error,
    }


def check_changes():
    plan = get_update_plan()
    if plan["mode"] != "noop":
        return True

    # Portable manifest metadata enrichment is allowed when content/index
    # identity is unchanged; it does not touch Qdrant or BM25.
    if plan["new_manifest"] != plan["old_manifest"]:
        try:
            ManifestManager.save(plan["new_manifest"])
        except Exception:
            pass
    return False


def smart_build():
    """Safely rebuild the production KB, returning False instead of crashing UI."""
    try:
        plan = get_update_plan()
        changes = plan["changes"]

        print("\n===== DOCUBOT PRODUCTION KB UPDATE =====")
        print(f"Source root      : {DOCUMENT_DIR}")
        print(f"Mode             : {plan['mode']}")
        print(f"Documents found  : {len(plan['documents'])}")
        print(f"Added            : {len(changes['added'])}")
        print(f"Updated          : {len(changes['updated'])}")
        print(f"Deleted          : {len(changes['deleted'])}")
        print(f"Unchanged        : {len(changes['unchanged'])}")

        if plan["mode"] == "noop":
            print("No production document/index changes detected.")
            return True

        preflight = get_kb_update_preflight(plan["documents"])
        if not preflight["ok"]:
            print("[FAIL] KB update preflight did not pass. Previous index was preserved.")
            for issue in preflight["issues"]:
                print(f"- {issue}")
            log_path = _write_failure_log("preflight", details=preflight["issues"])
            print(f"Diagnostic log: {log_path}")
            return False

        for note in preflight["notes"]:
            print(f"[INFO] {note}")

        if plan.get("reason"):
            print(f"Reason           : {plan['reason']}")
        print("Action           : transactional full Qdrant v4 rebuild")

        try:
            from scripts.build_qdrant_index import build_qdrant_index
        except Exception as error:
            log_path = _write_failure_log("build-module import", error=error)
            print("[FAIL] KB update could not start. Previous index was preserved.")
            print(f"Diagnostic log: {log_path}")
            return False

        if not build_qdrant_index():
            log_path = _write_failure_log(
                "transactional rebuild",
                details="build_qdrant_index returned False; review console output above.",
            )
            print(f"Diagnostic log: {log_path}")
            return False

        post_plan = get_update_plan()
        if post_plan["mode"] != "noop":
            details = post_plan.get("reason") or "Committed index did not pass post-check."
            log_path = _write_failure_log("post-build health", details=details)
            print("[FAIL] Rebuild finished but the committed KB did not pass the post-check.")
            print(f"Diagnostic log: {log_path}")
            return False

        print("[PASS] Production Qdrant v4 + BM25 update completed successfully.")
        return True

    except Exception as error:
        log_path = _write_failure_log("unexpected smart_build failure", error=error)
        print("[FAIL] KB update stopped safely. Previous working index was preserved.")
        print(f"Diagnostic log: {log_path}")
        return False


if __name__ == "__main__":
    raise SystemExit(0 if smart_build() else 1)
