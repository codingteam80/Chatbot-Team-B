from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HASH_MANIFEST = ROOT / "qa" / "v6_5_6_preserved_runtime_hashes.json"
RELEASE_MANIFEST = ROOT / "FINAL_COMPONENT_MANIFEST_v6.5.6.json"
BANK = ROOT / "qa" / "v6_5_6_quality_benchmark_bank.json"
OUT_DIR = ROOT / "logs" / "v6_5_6_quality_certification"

checks: list[dict] = []


def add(name: str, ok: bool, detail="") -> None:
    checks.append({"name": name, "pass": bool(ok), "detail": str(detail)})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8-sig")


try:
    runtime = json.loads(HASH_MANIFEST.read_text(encoding="utf-8"))
    runtime_files = runtime.get("files") or {}
    add("Runtime preservation hash manifest exists", bool(runtime_files), len(runtime_files))
    mismatches = []
    missing = []
    for rel, expected in runtime_files.items():
        path = ROOT / rel
        if not path.is_file():
            missing.append(rel)
            continue
        actual = sha256(path)
        if actual != expected:
            mismatches.append(rel)
    add("All production runtime files preserved from exact v6.5.5.4 baseline", not missing and not mismatches, f"missing={missing}; mismatches={mismatches}")
except Exception as error:
    add("Runtime preservation hash manifest parses", False, f"{type(error).__name__}: {error}")

try:
    manifest = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
    add("v6.5.6 release manifest exists/parses", True)
    add("Release version is v6.5.6", manifest.get("version") == "v6.5.6", manifest.get("version"))
    add("Generation model preserved", manifest.get("generation_model") == "qwen2.5:7b", manifest.get("generation_model"))
    add("Embedding model preserved", manifest.get("embedding_model") == "qwen3-embedding:8b", manifest.get("embedding_model"))
    add("Reranker preserved", manifest.get("reranker_model") == "BAAI/bge-reranker-v2-m3", manifest.get("reranker_model"))
    add("Chunk size preserved at 900", manifest.get("chunk_size") == 900, manifest.get("chunk_size"))
    add("Chunk overlap preserved at 150", manifest.get("chunk_overlap") == 150, manifest.get("chunk_overlap"))
    add("Retrieval threshold preserved at 0.55", float(manifest.get("minimum_retrieval_score")) == 0.55, manifest.get("minimum_retrieval_score"))
    topk = manifest.get("top_k") or {}
    add("Top-K preserved at vector/BM25/final 10/10/3", topk == {"vector": 10, "bm25": 10, "final": 3}, topk)
    mq = manifest.get("multi_query") or {}
    add("MultiQuery original + exactly two alternatives preserved", bool(mq.get("original_retained")) and mq.get("alternative_count") == 2, mq)
    qc = manifest.get("quality_certification") or {}
    add("v6.5.6 declares diagnostics-only production preservation", qc.get("production_runtime_changed") is False, qc)
    add("No production phrase-to-Rule hardcoding declared", qc.get("production_phrase_to_rule_hardcoding_added") is False, qc)
except Exception as error:
    add("v6.5.6 release manifest parses", False, f"{type(error).__name__}: {error}")

try:
    bank = json.loads(BANK.read_text(encoding="utf-8"))
    rows = list(bank.get("cases") or [])
    by_concept = {}
    for row in rows:
        by_concept.setdefault(str(row.get("concept_id") or ""), []).append(row)
    paired = {k: v for k, v in by_concept.items() if k and len(v) >= 2}
    add("Benchmark bank contains same-meaning concept pairs", len(paired) >= 6, f"pairs={len(paired)} cases={len(rows)}")
    add("Every benchmark case has expected reference", bool(rows) and all(row.get("expected_reference") for row in rows), len(rows))
    add("Benchmark bank is explicitly test-only", "Never imported by production runtime" in str(bank.get("purpose") or ""), bank.get("purpose"))
except Exception as error:
    add("Benchmark bank parses", False, f"{type(error).__name__}: {error}")

required = [
    "scripts/generate_v6_5_6_quality_benchmark.py",
    "scripts/test_v6_5_6_retrieval_quality.py",
    "scripts/test_v6_5_6_answer_quality.py",
    "scripts/test_v6_5_6_delete_lifecycle_live.py",
    "scripts/run_v6_5_6_quality_certification.py",
    "Run_v6.5.6_Quality_Certification.bat",
    "docs/QUALITY_CERTIFICATION_v6.5.6.txt",
    "QUALITY_CERTIFICATION_APPLIED_v6.5.6.txt",
]
for rel in required:
    add(f"Required v6.5.6 component exists: {rel}", (ROOT / rel).is_file())

try:
    retrieval = text("scripts/test_v6_5_6_retrieval_quality.py")
    answer = text("scripts/test_v6_5_6_answer_quality.py")
    lifecycle = text("scripts/test_v6_5_6_delete_lifecycle_live.py")
    runner = text("scripts/run_v6_5_6_quality_certification.py")
    add("Retrieval test reports Recall@3/Precision@3/Top-1", all(token in retrieval for token in ("recall_at_3", "precision_at_3", "top1_pass")))
    add("Retrieval test verifies semantic concept consistency", "concept_consistency" in retrieval and "top_reference_consistent" in retrieval)
    add("Retrieval test exposes MultiQuery used/early-accept decision", all(token in retrieval for token in ("multi_query_used", "multi_query_skipped_single_query_proven")))
    add("Retrieval test enforces exactly two alternatives when MQ is used", "alternative_count == 2" in retrieval)
    add("Answer diagnostic requires verified retrieval first", "retrieval/precision/top-1 gate failed" in answer)
    add("Answer diagnostic uses production deterministic claim-grounding guard", "validate_generated_claims" in answer and "claim_grounding_pass" in answer)
    add("Delete lifecycle requires clean NOOP baseline", "clean noop baseline" in lifecycle.casefold())
    add("Delete lifecycle checks manifest/Qdrant/BM25 probe removal", all(token in lifecycle for token in ("manifest_probe_present", "qdrant_probe_count", "bm25_probe_count")))
    add("Delete lifecycle protects semantic Rule-index hash", "semantic_index_hashes" in lifecycle and "semantic Rule index hash changed" in lifecycle)
    add("Delete lifecycle has fail-safe finally/recovery", "finally:" in lifecycle and "delete_lifecycle_recovery" in lifecycle)
    add("Certification runner includes live delete by default", "--skip-live-delete" in runner and "scripts.test_v6_5_6_delete_lifecycle_live" in runner)
    add("Certification runner uses 25-second normal-question target", "TARGET_SECONDS = 25.0" in runner)
    add("Certification runner creates evidence ZIP and SHA256", "ZipFile" in runner and ".sha256.txt" in runner)
except Exception as error:
    add("v6.5.6 diagnostic source inspection", False, f"{type(error).__name__}: {error}")

runtime_dirs = ["chains", "chat", "config", "embeddings", "ingestion", "llm", "retrieval", "runtime", "services", "ui", "utils"]
needle = "v6_5_6_quality_benchmark_bank"
hits = []
for folder in runtime_dirs:
    for path in (ROOT / folder).rglob("*.py"):
        try:
            if needle in path.read_text(encoding="utf-8-sig"):
                hits.append(path.relative_to(ROOT).as_posix())
        except Exception:
            pass
add("Production runtime does not import benchmark expectation bank", not hits, hits)

failed = [item for item in checks if not item["pass"]]
report = {
    "version": "v6.5.6",
    "validation": "PASS" if not failed else "FAIL",
    "checks": checks,
    "passed": len(checks) - len(failed),
    "failed": len(failed),
}
OUT_DIR.mkdir(parents=True, exist_ok=True)
out = OUT_DIR / "validation_latest.json"
out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"v6.5.6 quality certification validation: {report['validation']} ({report['passed']}/{len(checks)} PASS)")
for item in failed:
    print("[FAIL]", item["name"], item.get("detail", ""))
print(out)
raise SystemExit(0 if not failed else 1)
