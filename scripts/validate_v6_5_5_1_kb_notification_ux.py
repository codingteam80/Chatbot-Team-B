from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
MANIFEST = ROOT / "FINAL_COMPONENT_MANIFEST_v6.5.5.1.json"
OUTDIR = ROOT / "logs" / "v6_5_5_1_notification_ux"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "validation_latest.json"

checks = []

def check(name, passed, detail=""):
    checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})

source = APP.read_text(encoding="utf-8")
try:
    compile(source, str(APP), "exec")
    check("Python syntax: app.py", True)
except Exception as e:
    check("Python syntax: app.py", False, f"{type(e).__name__}: {e}")

# Extract only the pure helper so validation never starts Streamlit.
try:
    tree = ast.parse(source)
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_kb_update_success_message"
    )
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {}
    exec(compile(module, str(APP), "exec"), ns, ns)
    helper = ns["_kb_update_success_message"]
    check("Success notification helper exists and is isolated", True)
except Exception as e:
    helper = None
    check("Success notification helper exists and is isolated", False, f"{type(e).__name__}: {e}")

if helper:
    cases = [
        (
            {"plan": {"changes": {"added": [], "updated": ["a.txt"], "deleted": [], "unchanged": ["misra.pdf"]}}},
            "Knowledge Base is successfully updated.\n\nAdded: 0 · Updated: 1 · Deleted: 0",
        ),
        (
            {"plan": {"changes": {"added": ["a.txt", "b.pdf"], "updated": [], "deleted": ["old.docx"], "unchanged": []}}},
            "Knowledge Base is successfully updated.\n\nAdded: 2 · Updated: 0 · Deleted: 1",
        ),
        (
            {"plan": {"changes": {}}},
            "Knowledge Base is successfully updated.\n\nAdded: 0 · Updated: 0 · Deleted: 0",
        ),
    ]
    for i, (payload, expected) in enumerate(cases, 1):
        actual = helper(payload)
        check(f"Success notification exact output case {i}", actual == expected, repr(actual))

helper_start = source.find("def _kb_update_success_message")
helper_end = source.find("\n\n# ======================================\n# INITIALIZE", helper_start)
helper_source = source[helper_start:helper_end] if helper_start >= 0 and helper_end > helper_start else ""
check("Notification uses Added count", 'Added: {added}' in helper_source)
check("Notification uses Updated count", 'Updated: {updated}' in helper_source)
check("Notification uses Deleted count", 'Deleted: {deleted}' in helper_source)
for forbidden in (
    "changed chunks embedded",
    "Unchanged skipped",
    "Qdrant/BM25",
    "semantic Rule index",
    "KB Health all passed",
):
    check(f"User toast hides technical detail: {forbidden}", forbidden not in helper_source)

# Guard the surrounding UI behavior that was already fixed in v6.5.5.
check("Toast remains overlay-only", "st.toast(flash_message)" in source)
check("Maintenance panel remains stable keyed region", 'st.container(key="kb_maintenance_panel")' in source)
check("KB worker remains deferred", "kb_update_requested" in source and "launch_kb_update_subprocess" in source)
check("Confirmation checkbox remains configured", "kb_smart_update_confirm" in source)

try:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    check("v6.5.5.1 manifest exists and parses", True)
    check("Manifest version is v6.5.5.1", manifest.get("version") == "v6.5.5.1", manifest.get("version"))
    check("Models unchanged", manifest.get("generation_model") == "qwen2.5:7b" and manifest.get("embedding_model") == "qwen3-embedding:8b" and manifest.get("reranker_model") == "BAAI/bge-reranker-v2-m3")
    check("Chunking unchanged", manifest.get("chunk_size") == 900 and manifest.get("chunk_overlap") == 150)
    check("Threshold unchanged", manifest.get("minimum_retrieval_score") == 0.55)
    check("Top-K unchanged", manifest.get("top_k") == {"vector":10,"bm25":10,"final":3})
    check("Exactly two MultiQuery alternatives preserved", (manifest.get("multi_query") or {}).get("alternative_count") == 2)
    check("Incremental KB core explicitly unchanged", (manifest.get("kb_update") or {}).get("incremental_core_changed") is False)
    check("Notification policy is counts only", (manifest.get("ui") or {}).get("kb_update_success_notification") == "concise_counts_only")
except Exception as e:
    check("v6.5.5.1 manifest exists and parses", False, f"{type(e).__name__}: {e}")

failed = [c for c in checks if not c["passed"]]
report = {
    "version": "v6.5.5.1",
    "validation": "PASS" if not failed else "FAIL",
    "failed_checks": len(failed),
    "check_count": len(checks),
    "checks": checks,
}
OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"v6.5.5.1 notification UX validation: {report['validation']} ({len(checks)-len(failed)}/{len(checks)} PASS)")
for c in failed:
    print("[FAIL]", c["name"], c["detail"])
raise SystemExit(0 if not failed else 1)
