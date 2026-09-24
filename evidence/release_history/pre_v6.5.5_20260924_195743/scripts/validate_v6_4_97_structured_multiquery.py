from __future__ import annotations

import importlib
import json
import py_compile
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "logs" / "structured_multiquery"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "v6.4.97_structured_multiquery_validation_latest.json"

checks = []


def check(name, passed, detail=""):
    item = {"name": name, "passed": bool(passed), "detail": str(detail or "")}
    checks.append(item)
    print(("[PASS] " if passed else "[FAIL] ") + name + (f" :: {detail}" if detail else ""))


def text(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


for rel in [
    "config/settings.py",
    "config/prompts.py",
    "retrieval/multi_query.py",
    "retrieval/retriever.py",
    "llm/ollama_client.py",
]:
    try:
        py_compile.compile(str(ROOT / rel), doraise=True)
        check(f"Python syntax: {rel}", True)
    except Exception as error:
        check(f"Python syntax: {rel}", False, error)

settings_src = text("config/settings.py")
prompts_src = text("config/prompts.py")
mq_src = text("retrieval/multi_query.py")
retriever_src = text("retrieval/retriever.py")
ollama_src = text("llm/ollama_client.py")

check("Generation model default unchanged", '"qwen2.5:7b"' in settings_src)
check("Embedding model default unchanged", '"qwen3-embedding:8b"' in settings_src)
check("Reranker model default unchanged", '"BAAI/bge-reranker-v2-m3"' in settings_src)
check("Vector Top-K remains 10", bool(re.search(r"^VECTOR_TOP_K\s*=\s*10\s*$", settings_src, re.M)))
check("BM25 Top-K remains 10", bool(re.search(r"^BM25_TOP_K\s*=\s*10\s*$", settings_src, re.M)))
check("Final Top-K remains 3", bool(re.search(r"^FINAL_TOP_K\s*=\s*3\s*$", settings_src, re.M)))
check("Minimum retrieval score remains 0.55", bool(re.search(r"^MIN_RETRIEVAL_SCORE\s*=\s*0\.55\s*$", settings_src, re.M)))
check("Chunk size remains 900", bool(re.search(r"^CHUNK_SIZE\s*=\s*900\s*$", settings_src, re.M)))
check("Chunk overlap remains 150", bool(re.search(r"^CHUNK_OVERLAP\s*=\s*150\s*$", settings_src, re.M)))
check("MultiQuery remains enabled by default", 'MULTI_QUERY_RETRIEVAL_ENABLED = _env_bool("DOCUBOT_MULTI_QUERY", True)' in settings_src)
check("MultiQuery still defaults to two alternatives", 'str(min(3, int(os.getenv("DOCUBOT_MULTI_QUERY_VARIANTS", "2"))))' not in settings_src and '"DOCUBOT_MULTI_QUERY_VARIANTS", "2"' in settings_src)
check("RRF K remains 60", '"DOCUBOT_MULTI_QUERY_RRF_K", "60"' in settings_src)
check("Structured MultiQuery output enabled by default", '"DOCUBOT_MULTI_QUERY_STRUCTURED_OUTPUT"' in settings_src and 'MULTI_QUERY_STRUCTURED_OUTPUT_ENABLED = _env_bool(' in settings_src and 'True,' in settings_src)
check("Few-shot prompting remains available", 'FEW_SHOT_PROMPTING_ENABLED' in settings_src)
check("Silent reasoning remains available", 'INTERNAL_REASONING_GUIDANCE_ENABLED' in settings_src)

check("MultiQuery prompt requests JSON-only queries object", 'Return only a JSON object with one key named `queries`' in prompts_src)
check("MultiQuery prompt preserves dates and operators", 'dates, and technical operators such as' in prompts_src and '&&/||' in prompts_src)
check("MultiQuery few-shots use structured JSON examples", 'Good structured output: {{"queries"' in prompts_src)
check("Few-shots remain behavior-only", 'BEHAVIOR-ONLY MULTIQUERY EXAMPLES' in prompts_src and 'They are not answers' in prompts_src)

try:
    settings = importlib.import_module("config.settings")
    prompts = importlib.import_module("config.prompts")
    mq = importlib.import_module("retrieval.multi_query")
    check("Runtime structured-output default is ON", bool(settings.MULTI_QUERY_STRUCTURED_OUTPUT_ENABLED))
    check("Runtime alternative count is 2", int(settings.MULTI_QUERY_VARIANT_COUNT) == 2, settings.MULTI_QUERY_VARIANT_COUNT)

    schema = mq.multi_query_json_schema()
    qschema = schema.get("properties", {}).get("queries", {})
    check("JSON schema requires exactly two query strings", qschema.get("minItems") == 2 and qschema.get("maxItems") == 2 and qschema.get("items", {}).get("type") == "string")
    check("JSON schema forbids extra keys", schema.get("additionalProperties") is False and schema.get("required") == ["queries"])

    formatted = prompts.MULTI_QUERY_RETRIEVAL_PROMPT.format(variant_count=2, question="Pwede ba recursion sa MISRA?")
    check("Prompt formats cleanly with few-shot JSON braces", '"queries"' in formatted and "Pwede ba recursion sa MISRA?" in formatted)

    parsed = mq.parse_multi_query_variants(
        '{"queries":["requirements for labels referenced by goto statements","constraints on goto target label scope"]}',
        "What restrictions apply to goto labels?",
    )
    check("Structured JSON parser accepts two safe alternatives", len(parsed) == 2, parsed)

    guarded = mq.parse_multi_query_variants(
        '{"queries":["MISRA Rule 15.3 goto labels","constraints on goto target label scope"]}',
        "What restrictions apply to goto labels?",
    )
    check("Structured parser blocks invented Rule identifiers", guarded == ["constraints on goto target label scope"], guarded)

    legacy = mq.parse_multi_query_variants(
        "requirements for labels referenced by goto statements\nconstraints on goto target label scope",
        "What restrictions apply to goto labels?",
    )
    check("Guarded legacy text fallback remains available", len(legacy) == 2, legacy)

    class FakeClient:
        def __init__(self):
            self.schema = None
            self.fallback_called = False
        def generate_structured_json(self, prompt, schema):
            self.schema = schema
            return '{"queries":["requirements for labels referenced by goto statements","constraints on goto target label scope"]}'
        def generate(self, prompt):
            self.fallback_called = True
            raise AssertionError("legacy fallback should not be used on a valid structured response")

    fake = FakeClient()
    generated = mq.generate_multi_query_variants(fake, "What restrictions apply to goto labels?")
    check("MultiQuery generation prefers structured JSON path", len(generated) == 2 and not fake.fallback_called and fake.schema == schema, generated)
except Exception as error:
    check("Structured MultiQuery runtime unit checks", False, repr(error))

check("Ollama client exposes structured JSON generation", "def generate_structured_json" in ollama_src)
check("Structured call uses Ollama /api/chat", '/api/chat' in ollama_src)
check("Structured call sends JSON schema as format", '"format": dict(schema or {})' in ollama_src)
check("Structured call uses temperature 0", '"options": {"temperature": 0}' in ollama_src)
check("Structured call requests non-thinking output", '"think": False' in ollama_src)
check("Structured call is non-streaming", '"stream": False' in ollama_src)
check("Older Ollama think-field compatibility retry exists", 'compatibility_payload.pop("think", None)' in ollama_src)

try:
    oc = importlib.import_module("llm.ollama_client")
    captured = {}

    class FakeResponse:
        status_code = 200
        def raise_for_status(self):
            return None
        def json(self):
            return {"message": {"content": '{"queries":["alpha","beta"]}'}}

    original_post = oc.requests.post
    oc.requests.post = lambda url, **kwargs: (captured.update({"url": url, **kwargs}) or FakeResponse())
    try:
        client = oc.OllamaClient("qwen2.5:7b")
        client._wait_for_fast_recovery = lambda: None
        sample_schema = {
            "type": "object",
            "properties": {"queries": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2}},
            "required": ["queries"],
            "additionalProperties": False,
        }
        result = client.generate_structured_json("rewrite this", sample_schema)
    finally:
        oc.requests.post = original_post

    body = captured.get("json", {})
    check("Structured HTTP call targets /api/chat", str(captured.get("url", "")).endswith("/api/chat"), captured.get("url"))
    check("Structured HTTP call carries exact schema", body.get("format") == sample_schema)
    check("Structured HTTP call has deterministic options", body.get("options") == {"temperature": 0} and body.get("stream") is False and body.get("think") is False)
    check("Structured HTTP response returns message.content", result == '{"queries":["alpha","beta"]}', result)
except Exception as error:
    check("Structured Ollama HTTP unit checks", False, repr(error))
check("Structured generation retains transient connection retry", '_is_transient_connection_error(error)' in ollama_src)

check("Retriever still keeps original query first", 'searches = [original]' in retriever_src)
check("Retriever still uses RRF fusion", 'def _multi_query_rrf_fuse' in retriever_src and 'MULTI_QUERY_RRF_K' in retriever_src)
check("Retriever still reranks fused candidates with original intent", 'rerank_query' in retriever_src or 'intent_query' in retriever_src)
check("Near-threshold structured rescue remains present", 'def _near_threshold_structured_rule_rescue' in retriever_src)

manifest_path = ROOT / "FINAL_COMPONENT_MANIFEST_v6.4.97.json"
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check("v6.4.97 manifest exists", manifest.get("version") == "v6.4.97")
    check("Manifest says no KB rebuild", manifest.get("kb_rebuild") is False)
    check("Manifest says structured output default ON", manifest.get("multi_query", {}).get("structured_output_enabled_by_default") is True)
    check("Manifest preserves 0.55 threshold", manifest.get("architecture_lock", {}).get("minimum_retrieval_score") == 0.55)
    check("Manifest preserves Top-K 10/10/3", [manifest.get("architecture_lock", {}).get(k) for k in ("vector_top_k", "bm25_top_k", "final_top_k")] == [10, 10, 3])
    check("Manifest preserves two alternatives", manifest.get("multi_query", {}).get("alternative_count") == 2)
    check("Manifest records temperature 0", manifest.get("multi_query", {}).get("temperature") == 0)
    check("Manifest records visible CoT remains off", manifest.get("prompt_strategy", {}).get("visible_chain_of_thought_requested") is False)
except Exception as error:
    check("Manifest parse", False, error)

failed = [item for item in checks if not item["passed"]]
result = {
    "version": "v6.4.97",
    "overall": "PASS" if not failed else "FAIL",
    "checks": checks,
    "failed_count": len(failed),
    "kb_rebuild_performed": False,
    "minimum_retrieval_score": 0.55,
    "multi_query_default": True,
    "multi_query_alternative_count": 2,
    "structured_output_default": True,
    "structured_output_format": "Ollama JSON schema",
    "next": "READY_FOR_FOCUSED_QUESTION_TEST" if not failed else "BLOCKED",
}
OUTFILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
print("\nValidation result:", result["overall"])
print("Failed checks:", len(failed))
print("Output:", OUTFILE)
raise SystemExit(0 if not failed else 1)
