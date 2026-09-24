from __future__ import annotations

import hashlib
import json
import py_compile
import re
from pathlib import Path

from retrieval.semantic_rule_resolver import extract_rule_profiles, _load_bm25_records

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "logs" / "mixed_goal_benchmark"
BANK = ROOT / "qa" / "v6_5_1_mixed_goal_benchmark_bank.json"
MANIFEST = ROOT / "FINAL_COMPONENT_MANIFEST_v6.5.1.json"

V650_HASHES = {
    "retrieval/semantic_rule_resolver.py": "8d2ec37757ea6bcd9b5a41f38a3d6cbb582d02b364c8682d2c1092f8f2946091",
    "retrieval/retriever.py": "1956a36c2d215ae659d80279b5bee8219cf22141ba7a5f80651beb03629d7db6",
    "services/answer_service.py": "9e0fe2244a806c1c7601aa0bec5734305e1341ef28fa8b627731fa84eb29b46e",
    "services/misra_compliance.py": "d4e009d4a65d42d0828bb1956f97387c11defe11829ad9729ed455381d1b21d9",
    "embeddings/ollama_embedding.py": "e7259ca3ff7f2b0d37bd386a2c76e80a061511bd502b327b99b5f00b52806403",
    "runtime/prewarm.py": "39005b2bed2d1253abcd84633d3a62d81e53f836bf44db19629fb24eac71fff6"
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    checks = []

    def check(name: str, passed: bool, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail or "")})

    for rel in [
        "scripts/generate_v6_5_1_mixed_goal_benchmark.py",
        "scripts/evaluate_v6_5_1_mixed_goal_benchmark.py",
        "scripts/validate_v6_5_1_mixed_goal_benchmark.py",
    ]:
        try:
            py_compile.compile(str(ROOT / rel), doraise=True)
            check(f"Python syntax: {rel}", True)
        except Exception as error:
            check(f"Python syntax: {rel}", False, error)

    try:
        bank = json.loads(BANK.read_text(encoding="utf-8"))
        cases = list(bank.get("cases") or [])
    except Exception as error:
        bank, cases = {}, []
        check("Benchmark bank parses", False, error)
    else:
        check("Benchmark bank parses", True)

    questions = [str(item.get("question", "")).strip() for item in cases]
    concepts = {}
    for item in cases:
        concepts.setdefault(str(item.get("concept_id", "")), []).append(item)
    check("Benchmark bank has 24 questions", len(cases) == 24, len(cases))
    check("Benchmark bank has 12 concepts", len(concepts) == 12, len(concepts))
    check("Each benchmark concept has two natural variants", all(len(items) == 2 for items in concepts.values()))
    check("Benchmark questions are unique", len(set(q.casefold() for q in questions)) == len(questions))
    check("Benchmark questions are non-empty natural prompts", all(len(re.findall(r"\w+", q)) >= 8 for q in questions))

    registry_text = (ROOT / "qa" / "test_case_registry.py").read_text(encoding="utf-8", errors="replace")
    production_text = "\n".join(
        (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for rel in ["retrieval/semantic_rule_resolver.py", "retrieval/retriever.py", "services/answer_service.py", "services/misra_compliance.py"]
    )
    check("Benchmark questions are not registered in production QA", all(q not in registry_text for q in questions))
    check("Benchmark questions are absent from production runtime", all(q not in production_text for q in questions))

    profiles = extract_rule_profiles(_load_bm25_records())
    available = {profile.display_name for profile in profiles}
    expected = {str(item.get("expected_reference", "")).strip() for item in cases}
    check("All expected benchmark references exist in corpus-derived profiles", expected.issubset(available), sorted(expected - available))
    check("Corpus profile inventory remains 173", len(profiles) == 173, len(profiles))

    for rel, wanted in V650_HASHES.items():
        path = ROOT / rel
        actual = sha256(path) if path.is_file() else ""
        check(f"v6.5.0 production runtime unchanged: {rel}", actual == wanted, actual)

    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception as error:
        manifest = {}
        check("v6.5.1 manifest parses", False, error)
    else:
        check("v6.5.1 manifest parses", True)
    check("Manifest base is v6.5.0", manifest.get("base") == "v6.5.0")
    check("Manifest marks test-harness-only", manifest.get("test_harness_only") is True)
    check("Manifest records production runtime unchanged", manifest.get("production_runtime_changed") is False)
    check("Manifest preserves threshold 0.55", float(manifest.get("minimum_retrieval_score", -1)) == 0.55)
    check("Manifest preserves Top-K 10/10/3", manifest.get("retrieval_top_k") == {"vector": 10, "bm25": 10, "final": 3})
    check("Manifest preserves original + two MultiQuery alternatives", int(manifest.get("multi_query_alternative_count", -1)) == 2)
    check("Manifest says no KB rebuild", manifest.get("kb_rebuild_performed") is False and manifest.get("production_qdrant_or_bm25_rebuild_required") is False)

    passed = all(item["passed"] for item in checks)
    payload = {
        "version": "v6.5.1",
        "validation": "PASS" if passed else "FAIL",
        "failed_checks": sum(not item["passed"] for item in checks),
        "check_count": len(checks),
        "profile_count": len(profiles),
        "benchmark_question_count": len(cases),
        "benchmark_concept_count": len(concepts),
        "checks": checks,
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "v6.5.1_mixed_goal_benchmark_validation_latest.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    for item in checks:
        print(("[PASS] " if item["passed"] else "[FAIL] ") + item["name"] + (f" :: {item['detail']}" if item["detail"] else ""))
    print(f"\nValidation: {payload['validation']} | checks={len(checks)} | failed={payload['failed_checks']}")
    print(f"Output: {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
