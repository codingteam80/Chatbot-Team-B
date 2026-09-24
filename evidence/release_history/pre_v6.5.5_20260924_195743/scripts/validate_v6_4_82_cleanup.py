from __future__ import annotations

import ast
import hashlib
import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _record(checks, name, ok, detail=""):
    checks.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def main() -> int:
    checks = []
    print("DocuBot v6.4.82 Cleanup / Finalization Validation")
    print("=" * 78)

    try:
        from config import settings
    except Exception as exc:
        _record(checks, "Import config.settings", False, f"{type(exc).__name__}: {exc}")
        return _finish(checks)

    expected = {
        "knowledge_profile": (settings.KNOWLEDGE_PROFILE, "technical"),
        "vector_backend": (settings.VECTOR_BACKEND, "qdrant"),
        "chunking_profile": (settings.CHUNKING_PROFILE, "v4"),
        "embedding_backend": (settings.EMBEDDING_BACKEND, "ollama"),
        "embedding_model": (settings.EMBED_MODEL_NAME, "qwen3-embedding:8b"),
        "fast_model": (settings.OLLAMA_FAST_MODEL, "qwen2.5:7b"),
        "complex_model": (settings.OLLAMA_COMPLEX_MODEL, "qwen2.5:7b"),
        "reranker": (settings.RERANKER_MODEL, "BAAI/bge-reranker-v2-m3"),
        "chunk_size": (settings.CHUNK_SIZE, 900),
        "chunk_overlap": (settings.CHUNK_OVERLAP, 150),
        "vector_top_k": (settings.VECTOR_TOP_K, 10),
        "bm25_top_k": (settings.BM25_TOP_K, 10),
        "final_top_k": (settings.FINAL_TOP_K, 3),
    }
    for name, (actual, wanted) in expected.items():
        _record(checks, f"Canonical setting: {name}", actual == wanted, f"actual={actual!r}")

    storage = Path(settings.OPTION_C_STORAGE_DIR)
    _record(
        checks,
        "Active storage is production v4",
        storage.name == "option_c_qwen3_qdrant_v4",
        str(storage),
    )
    _record(checks, "Qdrant directory exists", Path(settings.QDRANT_DIR).is_dir(), str(settings.QDRANT_DIR))
    _record(checks, "BM25 directory exists", Path(settings.BM25_DIR).is_dir(), str(settings.BM25_DIR))
    _record(checks, "v4 ready marker exists", Path(settings.V4_PRODUCTION_READY_MARKER).is_file(), str(settings.V4_PRODUCTION_READY_MARKER))

    manifest_path = Path(settings.METADATA_DIR) / "manifest.json"
    manifest_ok = False
    manifest_detail = str(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_ok = bool(manifest) and all(
            isinstance(info, dict)
            and info.get("embedding_model") == "qwen3-embedding:8b"
            and info.get("index_schema_version") == settings.INDEX_SCHEMA_VERSION
            for info in manifest.values()
        )
        manifest_detail += f" entries={len(manifest)}"
    except Exception as exc:
        manifest_detail += f" error={type(exc).__name__}: {exc}"
    _record(checks, "Production manifest matches Qwen3 + Chunking-v4", manifest_ok, manifest_detail)

    forbidden_paths = [
        ROOT / "retrieval" / "chroma_search.py",
        ROOT / "scripts" / "build_index.py",
        ROOT / "Start_DocuBot_Certified_Default.bat",
        ROOT / "Start_DocuBot_v3_Rollback.bat",
        ROOT / "Start_DocuBot_Instruct_Test.bat",
        ROOT / "Pull_Instruct_Test_Models.bat",
        ROOT / "storage" / "chroma_db",
        ROOT / "storage" / "option_c_qwen3_qdrant",
        ROOT / "storage" / "experiments",
        ROOT / "data" / "all_documents",
    ]
    for path in forbidden_paths:
        _record(checks, f"Legacy active path removed: {path.relative_to(ROOT)}", not path.exists())

    required_root_files = [
        ROOT / "run.py",
        ROOT / "README.md",
        ROOT / "FINAL_COMPONENT_MANIFEST_v6.4.82.json",
        ROOT / ".vscode" / "launch.json",
        ROOT / ".vscode" / "settings.json",
        ROOT / "Run_v6.4.82_Final_Local_Validation.bat",
    ]
    for path in required_root_files:
        _record(checks, f"Required finalized file exists: {path.relative_to(ROOT)}", path.is_file())

    # Scan only active runtime/config code, never evidence/history/logs/venv.
    scan_roots = [
        "config", "retrieval", "embeddings", "llm", "runtime", "services",
        "chains", "chat", "ui", "utils", "ingestion",
    ]
    active_files = [ROOT / "app.py", ROOT / "run.py", ROOT / "requirements.txt"]
    for name in scan_roots:
        active_files.extend((ROOT / name).rglob("*.py"))

    forbidden_tokens = {
        "llama3 model reference": "llama3",
        "chromadb runtime reference": "chromadb",
        "legacy E5 reference": "multilingual-e5-base",
        "legacy HF embedding adapter": "llama_index.embeddings.huggingface",
    }
    for label, token in forbidden_tokens.items():
        hits = []
        for path in active_files:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace").casefold()
            if token.casefold() in text:
                hits.append(str(path.relative_to(ROOT)))
        _record(checks, f"No {label} in active runtime", not hits, ", ".join(hits[:10]))

    # Compile every active Python module plus the small production maintenance script set.
    maintenance_scripts = [
        "build_qdrant_index.py", "kb_health.py", "lan_common.py",
        "lan_server_preflight.py", "rebuild_index.py", "run_lan_server.py",
        "smart_build.py", "update_server_kb.py", "validate_v6_4_82_cleanup.py",
    ]
    compile_targets = [p for p in active_files if p.suffix == ".py"]
    compile_targets.extend(ROOT / "scripts" / name for name in maintenance_scripts)
    compile_errors = []
    for path in sorted(set(compile_targets)):
        if not path.is_file():
            compile_errors.append(f"missing:{path.relative_to(ROOT)}")
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            compile_errors.append(f"{path.relative_to(ROOT)}:{type(exc).__name__}:{exc}")
    _record(checks, "Active Python syntax/compile", not compile_errors, "; ".join(compile_errors[:8]))

    # Ensure the production maintenance scripts do not import archived Chroma modules.
    import_errors = []
    for name in maintenance_scripts:
        path = ROOT / "scripts" / name
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            import_errors.append(f"{name}:parse:{exc}")
            continue
        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = ",".join(alias.name for alias in node.names)
            if "retrieval.chroma_search" in module or "scripts.build_index" in module or "chromadb" in module:
                import_errors.append(f"{name}:{module}")
    _record(checks, "Maintenance scripts are Qdrant-only", not import_errors, "; ".join(import_errors[:8]))

    # Evidence must remain present after cleanup.
    evidence_dirs = [
        ROOT / "logs" / "chunking_v4_ab",
        ROOT / "logs" / "chunking_v4_certification",
        ROOT / "logs" / "phase7_performance",
        ROOT / "logs" / "phase7_optimization1",
        ROOT / "logs" / "phase7_optimization2",
        ROOT / "logs" / "phase8_production_lock",
        ROOT / "logs" / "english_natural_benchmark",
        ROOT / "logs" / "option_c_technical_qa",
    ]
    for path in evidence_dirs:
        count = sum(1 for p in path.rglob("*") if p.is_file()) if path.is_dir() else 0
        _record(checks, f"Evidence preserved: {path.relative_to(ROOT)}", count > 0, f"files={count}")

    try:
        from scripts.kb_health import run_health_check
        health_rc = int(run_health_check())
        _record(checks, "Read-only production KB health", health_rc == 0, f"exit_code={health_rc}")
    except Exception as exc:
        _record(checks, "Read-only production KB health", False, f"{type(exc).__name__}: {exc}")

    return _finish(checks)


def _finish(checks) -> int:
    failed = [c for c in checks if c["status"] != "PASS"]
    result = {
        "version": "v6.4.82",
        "phase": "cleanup_finalization_local_validation",
        "overall": "PASS" if not failed else "FAIL",
        "checks": checks,
    }
    outdir = ROOT / "logs" / "cleanup_finalization"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "v6.4.82_cleanup_validation_latest.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=" * 78)
    print("OVERALL:", result["overall"])
    print("Report :", out)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
