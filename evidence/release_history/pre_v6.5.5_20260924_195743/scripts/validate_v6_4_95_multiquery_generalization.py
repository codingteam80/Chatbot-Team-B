from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "logs" / "multiquery_generalization"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "v6.4.95_multiquery_generalization_validation_latest.json"

checks = []


def check(name: str, passed: bool, detail: str = ""):
    checks.append({"name": name, "passed": bool(passed), "detail": str(detail or "")})
    print(("[PASS] " if passed else "[FAIL] ") + name + (f" — {detail}" if detail else ""))


def text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


changed = [
    "config/settings.py",
    "config/prompts.py",
    "retrieval/multi_query.py",
    "retrieval/retriever.py",
    "chat/query_normalizer.py",
    "services/answer_service.py",
    "services/misra_compliance.py",
]

for rel in changed:
    path = ROOT / rel
    try:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        check(f"Python syntax: {rel}", True)
    except Exception as exc:
        check(f"Python syntax: {rel}", False, str(exc))

settings = text("config/settings.py")
prompts = text("config/prompts.py")
multi = text("retrieval/multi_query.py")
retriever = text("retrieval/retriever.py")
normalizer = text("chat/query_normalizer.py")
answer = text("services/answer_service.py")
misra = text("services/misra_compliance.py")

# Architecture lock remains unchanged.
check("Generation model unchanged", 'OLLAMA_FAST_MODEL = os.getenv(' in settings and settings.count('"qwen2.5:7b"') >= 2 and 'OLLAMA_COMPLEX_MODEL = os.getenv(' in settings)
check("Embedding model unchanged", 'qwen3-embedding:8b' in settings)
check("Vector Top-K remains 10", re.search(r"(?m)^VECTOR_TOP_K\s*=\s*10\s*$", settings) is not None)
check("BM25 Top-K remains 10", re.search(r"(?m)^BM25_TOP_K\s*=\s*10\s*$", settings) is not None)
check("Final Top-K remains 3", re.search(r"(?m)^FINAL_TOP_K\s*=\s*3\s*$", settings) is not None)
check("Minimum retrieval score remains 0.55", re.search(r"(?m)^MIN_RETRIEVAL_SCORE\s*=\s*0\.55\s*$", settings) is not None)
check("Chunk size remains 900", re.search(r"(?m)^CHUNK_SIZE\s*=\s*900\s*$", settings) is not None)
check("Chunk overlap remains 150", re.search(r"(?m)^CHUNK_OVERLAP\s*=\s*150\s*$", settings) is not None)

# MultiQuery profile.
check("MultiQuery enabled by default", 'MULTI_QUERY_RETRIEVAL_ENABLED = _env_bool("DOCUBOT_MULTI_QUERY", True)' in settings)
check("MultiQuery uses two alternatives by default", 'DOCUBOT_MULTI_QUERY_VARIANTS", "2"' in settings and 'MULTI_QUERY_VARIANT_COUNT = 2' in settings)
check("RRF K defaults to 60", 'DOCUBOT_MULTI_QUERY_RRF_K", "60"' in settings and 'MULTI_QUERY_RRF_K = 60' in settings)
check("MultiQuery prompt preserves intent and operators", "Preserve the original intent" in prompts and "&&/||" in prompts)
check("MultiQuery parser blocks invented structured identifiers", "variant_ids.issubset(original_ids)" in multi)
check("MultiQuery parser rejects answer-like generations", "Reject obvious answer-like generations" in multi)
check("Retriever implements reciprocal-rank fusion", "def _multi_query_rrf_fuse" in retriever and "MULTI_QUERY_RRF_K" in retriever)
check("Retriever reranks fused candidates with original intent", "rerank_query = str(intent_query or query or \"\").strip()" in retriever)

try:
    pos_ood = retriever.index("identity_ood_fast_fail")
    pos_mq = retriever.index("multi_query_searches = self._generate_multi_query_searches")
    pos_vector = retriever.index("vector_result_sets = []", pos_mq)
    check("MultiQuery runs after low-cost OOD/fast paths", pos_ood < pos_mq < pos_vector)
except ValueError as exc:
    check("MultiQuery runs after low-cost OOD/fast paths", False, str(exc))

check("MultiQuery widens only the reranker candidate pool", "candidates[:14 if multi_query_active else 10]" in retriever)
check("Near-threshold structured rescue exists without lowering 0.55", "def _near_threshold_structured_rule_rescue" in retriever and "MIN_RETRIEVAL_SCORE) - 0.06" in retriever)

# Generalization hardening found by the unseen set.
check("Logical && preserved during normalization", 'query.replace("&&", " logical and operator ")' in normalizer)
check("Logical || preserved during normalization", 'query.replace("||", " logical or operator ")' in normalizer)

followup_start = answer.find("def _is_grounded_followup_candidate")
followup_end = answer.find("def _question_has_visible_code", followup_start)
followup_block = answer[followup_start:followup_end]
check("Bare why/bakit no longer auto-anchors follow-up", 'r"^(?:why|bakit|' not in followup_block)
check("Bare what-rule/ano-rule no longer auto-anchors follow-up", 'ano(?:ng)?\\s+(?:rule|directive)' not in followup_block and 'what\\s+(?:rule|directive)' not in followup_block)
check("Short relational follow-up requires deictic reference", "reference_signal\n            and len(clean.split()) <= 9" in followup_block)

check("Generic allowed/permitted no longer means AUTHORIZED ENTITY", 'r"\\b(?:authorized|authorization|allowed|permitted|' not in answer)
check("Who/which entity authorization remains supported", "which\\s+(?:person|people|role|roles|team|group|organization|entity|entities)" in answer)

check("Structured catalog has specificity-before-family logic", "specific_statement_ids" in retriever and "extra_specific_tokens" in retriever)
check("Permission polarity recognizes 'there should be no'", "there\\s+(?:shall|should|must)\\s+be\\s+no" in misra)
check("Omission questions can use explicit required-presence source", "absence_request" in misra and "required_presence_statement" in misra)

manifest = ROOT / "FINAL_COMPONENT_MANIFEST_v6.4.95.json"
check("v6.4.95 manifest exists", manifest.is_file())
if manifest.is_file():
    data = json.loads(manifest.read_text(encoding="utf-8"))
    check("Manifest says no KB rebuild", data.get("kb_rebuild") is False)
    check("Manifest says global threshold/Top-K unchanged", data.get("retrieval_threshold_or_topk_changed") is False)
    check("Manifest records MultiQuery enabled", bool((data.get("multi_query") or {}).get("production_enabled_by_default")))

failed = [item for item in checks if not item["passed"]]
result = {
    "version": "v6.4.95",
    "overall": "PASS" if not failed else "FAIL",
    "checks": checks,
    "failed_count": len(failed),
    "kb_rebuild_performed": False,
    "minimum_retrieval_score": 0.55,
    "multi_query_default": True,
    "multi_query_alternative_count": 2,
    "next": "READY_FOR_FOCUSED_MULTIQUERY_RETEST" if not failed else "BLOCKED",
}
OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
print("\nOverall:", result["overall"])
print("Validation JSON:", OUT)
raise SystemExit(0 if not failed else 1)
