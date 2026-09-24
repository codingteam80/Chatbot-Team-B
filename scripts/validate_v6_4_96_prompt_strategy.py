from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "logs" / "prompt_strategy"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "v6.4.96_prompt_strategy_validation_latest.json"

checks = []


def check(name: str, passed: bool, detail: str = ""):
    checks.append({"name": name, "passed": bool(passed), "detail": str(detail or "")})
    print(("[PASS] " if passed else "[FAIL] ") + name + (f" — {detail}" if detail else ""))


def text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


for rel in ["config/settings.py", "config/prompts.py"]:
    path = ROOT / rel
    try:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
        check(f"Python syntax: {rel}", True)
    except Exception as exc:
        check(f"Python syntax: {rel}", False, str(exc))

settings = text("config/settings.py")
prompts = text("config/prompts.py")

# Architecture lock.
check("Generation model unchanged", settings.count('"qwen2.5:7b"') >= 2)
check("Embedding model unchanged", "qwen3-embedding:8b" in settings)
check("Reranker model unchanged", "BAAI/bge-reranker-v2-m3" in settings)
check("Vector Top-K remains 10", re.search(r"(?m)^VECTOR_TOP_K\s*=\s*10\s*$", settings) is not None)
check("BM25 Top-K remains 10", re.search(r"(?m)^BM25_TOP_K\s*=\s*10\s*$", settings) is not None)
check("Final Top-K remains 3", re.search(r"(?m)^FINAL_TOP_K\s*=\s*3\s*$", settings) is not None)
check("Minimum retrieval score remains 0.55", re.search(r"(?m)^MIN_RETRIEVAL_SCORE\s*=\s*0\.55\s*$", settings) is not None)
check("Chunk size remains 900", re.search(r"(?m)^CHUNK_SIZE\s*=\s*900\s*$", settings) is not None)
check("Chunk overlap remains 150", re.search(r"(?m)^CHUNK_OVERLAP\s*=\s*150\s*$", settings) is not None)
check("MultiQuery remains enabled by default", 'MULTI_QUERY_RETRIEVAL_ENABLED = _env_bool("DOCUBOT_MULTI_QUERY", True)' in settings)
check("MultiQuery still uses two alternatives", 'DOCUBOT_MULTI_QUERY_VARIANTS", "2"' in settings)
check("RRF K remains 60", 'DOCUBOT_MULTI_QUERY_RRF_K", "60"' in settings)

# New prompt strategy.
check("Few-shot prompting enabled by default", 'FEW_SHOT_PROMPTING_ENABLED = _env_bool("DOCUBOT_FEW_SHOT_PROMPT", True)' in settings)
check("Reasoning guidance enabled by default", '"DOCUBOT_REASONING_GUIDANCE"' in settings and "INTERNAL_REASONING_GUIDANCE_ENABLED" in settings)
check("Few-shot profile is behavior-only", 'PROMPT_FEW_SHOT_PROFILE = "behavior-only-v1"' in settings)

check("Silent evidence checklist exists", "SILENT EVIDENCE CHECK" in prompts)
check("Prompt explicitly forbids revealing chain-of-thought", "Do not reveal this checklist, chain-of-thought" in prompts)
check("General answer few-shots are marked non-factual", "These examples teach answer behavior only. They are NOT COMPANY KNOWLEDGE" in prompts)
check("MISRA few-shots are marked non-citable", "not MISRA facts and not citable evidence" in prompts)
check("MultiQuery few-shots are retrieval-only", "These examples teach retrieval-rewrite behavior only. They are not answers" in prompts)
check("MultiQuery few-shot preserves &&/||", "right operand of && or ||" in prompts and "logical &&/|| right operands" in prompts)
check("MultiQuery few-shot teaches Tagalog/English rewrite", "Pwede ba recursion sa MISRA?" in prompts)
check("Few-shots do not teach a concrete Rule number", re.search(r"Example[\s\S]{0,400}Rule\s+\d+\.\d+", prompts, re.I) is None)

# Runtime prompt activation default.
code = (
    "from config import settings;"
    "from config.prompts import SYSTEM_PROMPT,ANSWER_TEMPLATE,MISRA_COMPLIANCE_TEMPLATE,MULTI_QUERY_RETRIEVAL_PROMPT;"
    "print(int(settings.FEW_SHOT_PROMPTING_ENABLED));"
    "print(int(settings.INTERNAL_REASONING_GUIDANCE_ENABLED));"
    "print(int('SILENT EVIDENCE CHECK' in SYSTEM_PROMPT));"
    "print(int('BEHAVIOR-ONLY FEW-SHOT EXAMPLES' in ANSWER_TEMPLATE));"
    "print(int('MISRA BEHAVIOR-ONLY FEW-SHOT EXAMPLES' in MISRA_COMPLIANCE_TEMPLATE));"
    "print(int('BEHAVIOR-ONLY MULTIQUERY EXAMPLES' in MULTI_QUERY_RETRIEVAL_PROMPT));"
    "print(int('__DOCUBOT_' in SYSTEM_PROMPT+ANSWER_TEMPLATE+MISRA_COMPLIANCE_TEMPLATE+MULTI_QUERY_RETRIEVAL_PROMPT))"
)
env = dict(os.environ)
env["PYTHONPATH"] = str(ROOT)
p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, text=True, capture_output=True)
default_lines = [line.strip() for line in p.stdout.splitlines() if line.strip()]
check("Default runtime activates few-shot + silent reasoning", p.returncode == 0 and default_lines == ["1","1","1","1","1","1","0"], p.stderr.strip())

# A/B disable behavior.
env_off = dict(env)
env_off["DOCUBOT_FEW_SHOT_PROMPT"] = "0"
env_off["DOCUBOT_REASONING_GUIDANCE"] = "0"
p2 = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env_off, text=True, capture_output=True)
off_lines = [line.strip() for line in p2.stdout.splitlines() if line.strip()]
check("A/B switches cleanly remove optional prompt blocks", p2.returncode == 0 and off_lines == ["0","0","0","0","0","0","0"], p2.stderr.strip())

manifest = ROOT / "FINAL_COMPONENT_MANIFEST_v6.4.96.json"
check("v6.4.96 manifest exists", manifest.is_file())
if manifest.is_file():
    data = json.loads(manifest.read_text(encoding="utf-8"))
    check("Manifest says no KB rebuild", data.get("kb_rebuild") is False)
    check("Manifest says retrieval thresholds/Top-K unchanged", data.get("retrieval_threshold_or_topk_changed") is False)
    check("Manifest says MultiQuery algorithm unchanged", data.get("multi_query_algorithm_changed") is False)
    strategy = data.get("prompt_strategy") or {}
    check("Manifest records few-shot default ON", strategy.get("few_shot_enabled_by_default") is True)
    check("Manifest records visible CoT OFF", strategy.get("visible_chain_of_thought_requested") is False)

failed = [item for item in checks if not item["passed"]]
result = {
    "version": "v6.4.96",
    "overall": "PASS" if not failed else "FAIL",
    "checks": checks,
    "failed_count": len(failed),
    "kb_rebuild_performed": False,
    "minimum_retrieval_score": 0.55,
    "multi_query_default": True,
    "few_shot_prompting_default": True,
    "silent_reasoning_guidance_default": True,
    "visible_chain_of_thought": False,
    "next": "READY_FOR_FOCUSED_PROMPTING_RETEST" if not failed else "BLOCKED",
}
OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
print("\nOverall:", result["overall"])
print("Validation JSON:", OUT)
raise SystemExit(0 if not failed else 1)
