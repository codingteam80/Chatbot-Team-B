from __future__ import annotations

from pathlib import Path

from config.settings import EMBED_MODEL_NAME, INDEX_SCHEMA_VERSION
from retrieval.bm25_index import CORPUS_FILE, INDEX_FILE
from utils.file_utils import get_all_documents
from utils.manifest import ManifestManager


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


def get_update_plan():
    """Return the canonical production update plan.

    v6.4.82 has one active vector backend: Qdrant v4.  Any source-document
    change is therefore handled by the already-certified transactional full
    Qdrant rebuild.
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

    # Cheap manifest metadata enrichment is allowed when content/index identity
    # is unchanged; it does not touch Qdrant or BM25.
    if plan["new_manifest"] != plan["old_manifest"]:
        try:
            ManifestManager.save(plan["new_manifest"])
        except Exception:
            pass
    return False


def smart_build():
    plan = get_update_plan()
    changes = plan["changes"]

    print("\n===== DOCUBOT PRODUCTION KB UPDATE =====")
    print(f"Mode            : {plan['mode']}")
    print(f"Documents found : {len(plan['documents'])}")
    print(f"Added           : {len(changes['added'])}")
    print(f"Updated         : {len(changes['updated'])}")
    print(f"Deleted         : {len(changes['deleted'])}")
    print(f"Unchanged       : {len(changes['unchanged'])}")

    if plan["mode"] == "noop":
        print("No production document/index changes detected.")
        return True

    if plan.get("reason"):
        print(f"Reason          : {plan['reason']}")
    print("Action          : transactional full Qdrant v4 rebuild")

    from scripts.build_qdrant_index import build_qdrant_index

    if not build_qdrant_index():
        return False

    post_plan = get_update_plan()
    if post_plan["mode"] != "noop":
        print("[ERROR] Qdrant rebuild completed but the committed index is not healthy.")
        if post_plan.get("reason"):
            print(f"Reason: {post_plan['reason']}")
        return False

    print("Production Qdrant v4 + BM25 update completed successfully.")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if smart_build() else 1)
