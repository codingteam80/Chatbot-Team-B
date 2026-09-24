from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" :: {detail}" if detail else ""))
    return {"name": name, "status": status, "detail": str(detail)}


def main() -> int:
    checks = []

    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    checks.append(check(
        "requirements references current environment setup BAT",
        "Setup_DocuBot_Production_Environment.bat" in req
        and "Setup_DocuBot_Option_C.bat" not in req,
    ))

    bq = (ROOT / "scripts" / "build_qdrant_index.py").read_text(encoding="utf-8")
    checks.append(check(
        "Qdrant install diagnostic references current setup BAT",
        "Setup_DocuBot_Production_Environment.bat" in bq
        and "Setup_DocuBot_Option_C.bat" not in bq,
    ))

    for rel in [
        "Setup_DocuBot_Production_Environment.bat",
        "scripts/setup_production_environment.py",
        "FINAL_COMPONENT_MANIFEST_v6.4.83.json",
        "README.md",
        "docs/PRODUCTION_COMPONENTS.md",
    ]:
        checks.append(check(f"Required v6.4.83 file exists: {rel}", (ROOT / rel).is_file()))

    checks.append(check(
        "Superseded model-only setup BAT removed from active root",
        not (ROOT / "Setup_DocuBot_Production_Models.bat").exists(),
    ))
    checks.append(check(
        "Superseded v6.4.82 validation BAT removed from active root",
        not (ROOT / "Run_v6.4.82_Final_Local_Validation.bat").exists(),
    ))
    checks.append(check(
        "v6.4.82 component manifest moved out of active root",
        not (ROOT / "FINAL_COMPONENT_MANIFEST_v6.4.82.json").exists(),
    ))

    try:
        from config import settings
        expected = {
            "KNOWLEDGE_PROFILE": "technical",
            "VECTOR_BACKEND": "qdrant",
            "CHUNKING_PROFILE": "v4",
            "EMBEDDING_BACKEND": "ollama",
            "EMBED_MODEL_NAME": "qwen3-embedding:8b",
            "FAST_MODEL": "qwen2.5:7b",
            "COMPLEX_MODEL": "qwen2.5:7b",
            "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
            "CHUNK_SIZE": 900,
            "CHUNK_OVERLAP": 150,
            "VECTOR_TOP_K": 10,
            "BM25_TOP_K": 10,
            "FINAL_TOP_K": 3,
        }
        for attr, wanted in expected.items():
            actual = getattr(settings, attr)
            checks.append(check(f"Canonical setting: {attr}", actual == wanted, f"actual={actual!r}"))
    except Exception as exc:
        checks.append(check("Canonical settings import", False, f"{type(exc).__name__}: {exc}"))

    # Python source compile only; no network/model pull here.
    compile_targets = [
        ROOT / "scripts" / "setup_production_environment.py",
        ROOT / "scripts" / "build_qdrant_index.py",
    ]
    for p in compile_targets:
        try:
            compile(p.read_text(encoding="utf-8"), str(p), "exec")
            checks.append(check(f"Python syntax: {p.relative_to(ROOT)}", True))
        except Exception as exc:
            checks.append(check(f"Python syntax: {p.relative_to(ROOT)}", False, exc))

    setup_text = (ROOT / "scripts" / "setup_production_environment.py").read_text(encoding="utf-8")
    checks.append(check(
        "Setup pulls only canonical Ollama models",
        '("qwen2.5:7b", "qwen3-embedding:8b")' in setup_text
        and "llama3" not in setup_text.lower()
        and "chroma" not in setup_text.lower(),
    ))
    checks.append(check(
        "Setup does not invoke KB rebuild",
        "rebuild_index" not in setup_text
        and "build_qdrant_index" not in setup_text,
    ))

    overall = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    outdir = ROOT / "logs" / "deployment_bootstrap"
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "v6.4.83",
        "phase": "deployment_bootstrap_structural_validation",
        "overall": overall,
        "checks": checks,
    }
    (outdir / "v6.4.83_deployment_bootstrap_validation_latest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print()
    print("Overall:", overall)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
