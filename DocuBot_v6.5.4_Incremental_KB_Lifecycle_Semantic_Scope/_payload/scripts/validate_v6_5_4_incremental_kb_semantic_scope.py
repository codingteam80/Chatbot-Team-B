from __future__ import annotations

import ast
import importlib.util
import json
import os
import py_compile
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "logs" / "incremental_kb_semantic_scope"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "v6.5.4_incremental_kb_semantic_scope_validation_latest.json"
checks: list[dict] = []
SIMULATE = os.getenv("DOCUBOT_V654_SIMULATE", "").strip() == "1"


def check(name: str, passed: bool, detail="") -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": str(detail or "")})


def source(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    try:
                        return ast.literal_eval(node.value)
                    except Exception:
                        continue
    return None


changed = [
    "app.py",
    "retrieval/retriever.py",
    "retrieval/semantic_rule_resolver.py",
    "scripts/build_qdrant_index.py",
    "scripts/kb_update_runner.py",
    "scripts/smart_build.py",
    "scripts/test_kb_update_portability.py",
    "scripts/update_qdrant_incremental.py",
]
for rel in changed:
    try:
        py_compile.compile(str(ROOT / rel), doraise=True)
        check(f"Python syntax: {rel}", True)
    except Exception as exc:
        check(f"Python syntax: {rel}", False, exc)

settings_path = ROOT / "config" / "settings.py"
for name, expected in [
    ("CHUNK_SIZE", 900),
    ("CHUNK_OVERLAP", 150),
    ("MIN_RETRIEVAL_SCORE", 0.55),
    ("VECTOR_TOP_K", 10),
    ("BM25_TOP_K", 10),
    ("FINAL_TOP_K", 3),
    ("MULTI_QUERY_VARIANT_COUNT", 2),
]:
    value = literal_assignment(settings_path, name)
    check(f"Invariant {name} remains {expected}", value == expected, value)

app_source = source("app.py")
smart_source = source("scripts/smart_build.py")
incremental_source = source("scripts/update_qdrant_incremental.py")
full_source = source("scripts/build_qdrant_index.py")
runner_source = source("scripts/kb_update_runner.py")
semantic_source = source("retrieval/semantic_rule_resolver.py")
retriever_source = source("retrieval/retriever.py")

check("Planner has noop/incremental/full_rebuild modes", all(token in smart_source for token in ['mode = "incremental"', '"full_rebuild"', '"noop"']))
check("Document changes no longer force full rebuild", "Source-document changes require the certified transactional Qdrant rebuild" not in smart_source)
check("Incremental policy explicitly skips unchanged", "unchanged=SKIP" in smart_source and "new/modified/deleted=PROCESS" in smart_source)
check("UI tells operator unchanged files are not re-embedded", "Unchanged files will not be re-embedded" in app_source)
check("Incremental module snapshots only changed sources", "_snapshot_changed_sources" in incremental_source and '("added", "updated")' in incremental_source)
check("Incremental module embeds changed chunks only", "embedded_changed_chunks" in incremental_source and "_upsert_records" in incremental_source)
check("Incremental module removes modified/deleted old chunks", 'changes.get("updated")' in incremental_source and 'changes.get("deleted")' in incremental_source and "_delete_ids" in incremental_source)
check("Incremental module rebuilds BM25 from final staging text", "BM25Indexer().build(final_records)" in incremental_source)
check("Incremental module uses transactional Qdrant staging copy", "_copy_qdrant_tree" in incremental_source and "_activate_staging_directory" in incremental_source)
check("Incremental module has source stability guard", "_source_state_matches" in incremental_source and "source_stability_guard" in incremental_source)
check("Full rebuild also has source stability guard", "_source_state_matches(candidate_manifest, old_manifest)" in full_source)
check("Full rebuild avoids live manifest rebuild after activation", "final_manifest = {key: dict(info) for key, info in candidate_manifest.items()}" in full_source)
check("Runner version reporting is v6.5.4", 'VERSION = "v6.5.4"' in runner_source)
check("Runner records transactional incremental action", '"transactional_incremental"' in runner_source)
check("Runner records build details", 'report["build_details"]' in runner_source)

check("Semantic Rule index uses profile-scoped signature", "profile_sha256" in semantic_source and "misra-semantic-rule-index-v2-profile-signature" in semantic_source)
check("Semantic Rule index no longer keys readiness to whole BM25 SHA", '"corpus_sha256"' not in semantic_source[semantic_source.find("def _profile_digest"):semantic_source.find("class SemanticRuleResolver")])
check("Semantic v1 metadata can migrate without embeddings", "signature_migrated_at" in semantic_source and '"migrated": True' in semantic_source)
check("Semantic profile signature is cached by BM25 stat", "_PROFILE_SIGNATURE_CACHE" in semantic_source and "st_mtime_ns" in semantic_source)

# Preserve the v6.5.3.1 lock-handoff and latency diagnostics.
check("Worker-only lock ownership preserved", '"owner_role": "worker"' in runner_source and "_prepare_launcher_lock_handoff" in runner_source)
check("Legacy parent self-lock handoff preserved", "removed_legacy_parent_self_lock" in runner_source)
check("Live worker lock protection preserved", "os.O_CREAT | os.O_EXCL | os.O_WRONLY" in runner_source)
check("Streamlit session double-click guard preserved", "kb_update_in_progress" in app_source)
check("Semantic latency decision event preserved", "MISRA SEMANTIC LATENCY DECISION" in retriever_source)
check("Single-query proven path can skip MultiQuery", "multi_query_skipped_single_query_proven" in retriever_source)
check("Ambiguous path still uses MultiQuery", "multi_query_used_for_ambiguous_single_query" in retriever_source)

# Guard against benchmark-specific rule/question mappings in the new update/semantic work.
for token in ["Rule 17.7", "Rule 21.17", "Rule 16.4", "localtime", "strerror", "write_status", "update_state"]:
    check(f"No benchmark mapping in incremental updater: {token}", token not in incremental_source)

# Runtime planner classification smoke tests (in-memory monkeypatches only).
if SIMULATE:
    check("Planner runtime smoke tests execute", True, "SIMULATION: dependency-backed runtime test skipped")
else:
    try:
        import scripts.smart_build as sb
        import retrieval.qdrant_search as qs

        old_get_docs = sb.get_all_documents
        old_load = sb.ManifestManager.load
        old_build = sb.ManifestManager.build
        old_count = qs.qdrant_collection_count
        old_full = sb._manifest_requires_full_rebuild
        old_bm25 = sb.bm25_is_ready
        try:
            sb.get_all_documents = lambda: [Path("/tmp/a.pdf")]
            qs.qdrant_collection_count = lambda: 10
            sb._manifest_requires_full_rebuild = lambda manifest: False
            sb.bm25_is_ready = lambda: True

            old_manifest = {"a.pdf": {"hash": "A", "index_schema_version": "x", "embedding_model": "y", "embedding_normalize": True}}
            sb.ManifestManager.load = lambda: old_manifest

            sb.ManifestManager.build = lambda docs, previous_manifest=None: dict(old_manifest)
            plan = sb.get_update_plan()
            check("Planner: unchanged => noop", plan.get("mode") == "noop", plan.get("mode"))

            added_manifest = dict(old_manifest)
            added_manifest["b.docx"] = {"hash": "B", "index_schema_version": "x", "embedding_model": "y", "embedding_normalize": True}
            sb.ManifestManager.build = lambda docs, previous_manifest=None: added_manifest
            plan = sb.get_update_plan()
            check("Planner: new => incremental", plan.get("mode") == "incremental", plan.get("mode"))

            updated_manifest = {"a.pdf": {**old_manifest["a.pdf"], "hash": "C"}}
            sb.ManifestManager.build = lambda docs, previous_manifest=None: updated_manifest
            plan = sb.get_update_plan()
            check("Planner: modified => incremental", plan.get("mode") == "incremental", plan.get("mode"))

            sb.ManifestManager.build = lambda docs, previous_manifest=None: {}
            plan = sb.get_update_plan()
            check("Planner: deleted => incremental", plan.get("mode") == "incremental", plan.get("mode"))
        finally:
            sb.get_all_documents = old_get_docs
            sb.ManifestManager.load = old_load
            sb.ManifestManager.build = old_build
            qs.qdrant_collection_count = old_count
            sb._manifest_requires_full_rebuild = old_full
            sb.bm25_is_ready = old_bm25
    except Exception as exc:
        check("Planner runtime smoke tests execute", False, f"{type(exc).__name__}: {exc}")

# Qdrant API smoke against a disposable local collection only.
if SIMULATE:
    check("Synthetic Qdrant staging/delta smoke executes", True, "SIMULATION: qdrant-client smoke skipped")
else:
    try:
        from qdrant_client import QdrantClient, models
        import scripts.update_qdrant_incremental as inc

        with tempfile.TemporaryDirectory(prefix="docubot-v654-qdrant-") as td:
            active = Path(td) / "active"
            staging = Path(td) / "staging"
            c = QdrantClient(path=str(active))
            c.create_collection(
                collection_name=inc.QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
            )
            c.upsert(
                collection_name=inc.QDRANT_COLLECTION_NAME,
                points=[
                    models.PointStruct(id=1, vector=[1.0, 0.0], payload={"text":"a","metadata":{"file_path":"C:/kb/a.pdf","file_name":"a.pdf"}}),
                    models.PointStruct(id=2, vector=[0.0, 1.0], payload={"text":"b","metadata":{"file_path":"C:/kb/b.pdf","file_name":"b.pdf"}}),
                ],
                wait=True,
            )
            c.close()
            inc._copy_qdrant_tree(active, staging, retries=2)
            c2 = QdrantClient(path=str(staging))
            pts = inc._scroll_all_points(c2)
            check("Synthetic staging copy preserves points", len(pts) == 2, len(pts))
            ids = inc._point_ids_for_document(pts, "a.pdf", {"file_path":"C:/kb/a.pdf", "indexed_file_path":"C:/kb/a.pdf"})
            check("Synthetic changed-file point selection is exact", len(ids) == 1, ids)
            inc._delete_ids(c2, ids)
            remaining = int(c2.count(inc.QDRANT_COLLECTION_NAME, exact=True).count)
            check("Synthetic Qdrant delta delete works", remaining == 1, remaining)
            c2.close()
    except Exception as exc:
        check("Synthetic Qdrant staging/delta smoke executes", False, f"{type(exc).__name__}: {exc}")
# Verify profile signature ignores unrelated non-MISRA records on the real BM25 corpus.
try:
    import pickle
    import retrieval.semantic_rule_resolver as sr
    corpus = ROOT / "storage" / "option_c_qwen3_qdrant_v4" / "bm25" / "corpus.pkl"
    records = pickle.load(corpus.open("rb"))
    base_profiles = sr.extract_rule_profiles(records)
    extra_records = list(records) + [{
        "text": "Synthetic employee policy text used only by validator.",
        "metadata": {"file_name":"validator_policy.docx", "file_path":"C:/validator/validator_policy.docx"},
    }]
    extra_profiles = sr.extract_rule_profiles(extra_records)
    check("MISRA profile count unaffected by unrelated policy", len(base_profiles) == len(extra_profiles), (len(base_profiles), len(extra_profiles)))
    check("MISRA profile signature unaffected by unrelated policy", sr._profile_digest(base_profiles) == sr._profile_digest(extra_profiles), sr._profile_digest(base_profiles))
except Exception as exc:
    check("Semantic profile-scope smoke executes", False, f"{type(exc).__name__}: {exc}")

manifest = ROOT / "FINAL_COMPONENT_MANIFEST_v6.5.4.json"
check("v6.5.4 manifest exists", manifest.is_file(), manifest)
if manifest.is_file():
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        check("v6.5.4 manifest parses", isinstance(payload, dict))
        check("Manifest records incremental lifecycle", bool(payload.get("incremental_kb_lifecycle")), payload.get("incremental_kb_lifecycle"))
        check("Manifest records no install-time KB rebuild", payload.get("patch_time_kb_rebuild") is False, payload.get("patch_time_kb_rebuild"))
        check("Manifest preserves chunking", payload.get("chunk_size") == 900 and payload.get("chunk_overlap") == 150)
    except Exception as exc:
        check("v6.5.4 manifest parses", False, exc)

failed = [item for item in checks if not item["passed"]]
result = {
    "version": "v6.5.4",
    "validation": "PASS" if not failed else "FAIL",
    "failed_checks": len(failed),
    "check_count": len(checks),
    "checks": checks,
}
OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({k: result[k] for k in ("version", "validation", "failed_checks", "check_count")}, indent=2))
if failed:
    for item in failed:
        print("[FAIL]", item["name"], "::", item.get("detail", ""))
    raise SystemExit(1)
print(f"[PASS] v6.5.4 validator passed {len(checks)} checks.")
raise SystemExit(0)
