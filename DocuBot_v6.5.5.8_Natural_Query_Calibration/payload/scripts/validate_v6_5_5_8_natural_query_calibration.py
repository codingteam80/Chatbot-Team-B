from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Lightweight shims only for offline validation environments that do not have
# the production UI/BM25 packages installed. Production PCs use the real libs.
if importlib.util.find_spec("streamlit") is None:
    streamlit = types.ModuleType("streamlit")
    streamlit.cache_resource = lambda *a, **k: (lambda fn: fn)
    streamlit.session_state = {}
    sys.modules["streamlit"] = streamlit

if importlib.util.find_spec("rank_bm25") is None:
    rank_bm25 = types.ModuleType("rank_bm25")
    rank_bm25.BM25Okapi = type("BM25Okapi", (), {})
    sys.modules["rank_bm25"] = rank_bm25

from ingestion.pdf_structure import PDFStructureExtractor
from retrieval.retriever import CompanyRetriever
from services.answer_service import AnswerService
from services.misra_compliance import MisraComplianceMode
from utils.structured_reference import StructuredReference
from config.settings import BM25_DIR, TECHNICAL_DOCUMENT_DIR

checks: list[dict] = []


def check(name: str, ok: bool, detail=None) -> None:
    checks.append({"name": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail not in (None, "") else ""))


def refs_for(question: str) -> list[str]:
    results, _diag = MisraComplianceMode.rule_body_cue_rescue_from_authoritative_corpus(question, top_k=6)
    return [str((item.get("metadata", {}) or {}).get("section_title", "") or "") for item in results]


# Boundary regression: "externally visible" is C linkage vocabulary, not an
# instruction to leave the KB.
q_external_visibility = "I have a helper function used only inside one .c file. Should it still be externally visible?"
check("external-linkage wording stays inside KB", not AnswerService._explicit_external_knowledge_request(q_external_visibility))
check("explicit external-knowledge request still blocked", AnswerService._explicit_external_knowledge_request("Use external knowledge to answer this."))

# Source-language cue calibration. Expected Rule numbers live only in this QA
# validator, never in production routing code.
rule87_questions = [
    q_external_visibility,
    "Kung isang translation unit lang gumagamit ng function, ano ang MISRA guidance sa linkage nito?",
]
for idx, q in enumerate(rule87_questions, 1):
    check(f"Rule 8.7 natural cue {idx}", refs_for(q) == ["Rule 8.7"], refs_for(q))

rule106_questions = [
    "I add two uint16_t values and store the result in a uint32_t. Is the wider destination enough?",
    "Kung 16-bit operands ang addition pero 32-bit variable ang destination, may MISRA concern ba?",
    "Does assigning a narrow arithmetic expression to a wider variable make the calculation itself wider?",
]
for idx, q in enumerate(rule106_questions, 1):
    check(f"Rule 10.6 natural cue {idx}", refs_for(q) == ["Rule 10.6"], refs_for(q))

rule144_questions = [
    "Can I write if (counter) under MISRA?",
    "Kelangan bang explicit comparison ang integer sa condition ng if?",
    "A loop uses a pointer directly as its condition: while (ptr). Which rule is relevant?",
]
for idx, q in enumerate(rule144_questions, 1):
    cues = MisraComplianceMode.semantic_cues(q)
    check(f"Rule 14.4 natural cue {idx}", refs_for(q) == ["Rule 14.4"], refs_for(q))
    check(
        f"Rule 14.4 query {idx} does not inject Rule 15.6 compound-body cue",
        "the body of an iteration-statement or a selection-statement shall be a compound-statement" not in cues,
        cues,
    )

empty_else = "Can I have an empty else block?"
empty_cues = MisraComplianceMode.semantic_cues(empty_else)
check("empty-else query resolves to Rule 15.7 source cue", refs_for(empty_else) == ["Rule 15.7"], refs_for(empty_else))
check(
    "empty-else prose does not inject Rule 15.6",
    "the body of an iteration-statement or a selection-statement shall be a compound-statement" not in empty_cues,
    empty_cues,
)
state, observation = MisraComplianceMode.cue_assessment_state(empty_else, empty_cues[0])
check("generic empty-else question stays conditional", state == "uncertain", {"state": state, "observation": observation})

# Build a retriever shell around the active corpus. No vector model/reranker is
# loaded for these structured checks.
corpus_path = BM25_DIR / "corpus.pkl"
with corpus_path.open("rb") as handle:
    records = pickle.load(handle)
retriever = CompanyRetriever.__new__(CompanyRetriever)
retriever.bm25 = types.SimpleNamespace(records=records)
retriever._structured_pdf_family_cache = {}

# Generic list stop-word normalization must not turn "related" -> "relat" into
# an accidental concept match such as Rule 20.14.
goto_items = retriever._retrieve_structured_rule_catalog(
    "List rules related to goto statements.",
    "List rules related to goto statements.",
)
goto_ids = [str((item.get("metadata", {}) or {}).get("rule_id", "") or "") for item in goto_items]
check("goto inventory is complete and precise", goto_ids == ["15.1", "15.2", "15.3", "15.4"], goto_ids)

mandatory_items = retriever._retrieve_structured_rule_catalog(
    "List mandatory MISRA rules.",
    "List mandatory MISRA rules.",
)
mandatory_ids = [str((item.get("metadata", {}) or {}).get("rule_id", "") or "") for item in mandatory_items]
expected_mandatory = [
    "9.1", "12.5", "13.6", "17.3", "17.4", "17.6", "19.1",
    "21.13", "21.17", "21.18", "21.19", "21.20", "22.2", "22.4", "22.5", "22.6",
]
check("mandatory Rule inventory includes source-complete Rule 22.2", mandatory_ids == expected_mandatory, mandatory_ids)

# Existing active index has one historical split boundary at Rule 22.2. Exact
# runtime lookup must recover the full source block without forcing re-embedding.
exact_222 = retriever._retrieve_exact_structured_reference(StructuredReference("rule", "22.2"))
exact_text = str(exact_222[0].get("text", "") if exact_222 else "")
check("exact Rule 22.2 runtime source recovery", "Standard Library function" in exact_text and "Category\nMandatory" in exact_text, exact_text[:260])

# Parser fix is narrow: the source Rule must now remain one logical requirement.
pdf_path = TECHNICAL_DOCUMENT_DIR / "MISRA_FromInternet.pdf"
sections = PDFStructureExtractor.extract(str(pdf_path))
rule222_sections = [s for s in sections if str(s.section_title).strip() == "Rule 22.2"]
parsed_222 = rule222_sections[0].text if rule222_sections else ""
check("Rule 22.2 structure parser keeps wrapped requirement", "Standard Library function" in parsed_222 and "Category\nMandatory" in parsed_222, parsed_222[:260])

# List/family evidence must survive post-retrieval authoritative promotion.
preserve_result = AnswerService._post_retrieval_misra_authoritative_promotion(
    "List rules related to goto statements.",
    [{"_structured_topic_anchor": True, "text": "Rule 15.1", "metadata": {"rule_id": "15.1"}}],
)
check("structured list intent cannot collapse to one semantic Rule", preserve_result[0] == "" and preserve_result[2].get("preserved_structured_inventory") is True, preserve_result[2])

# Anti-overfit guard: exact calibration sentences must not appear in production
# source files. Tests may contain them; production routing may not.
production_files = [
    ROOT / "services" / "answer_service.py",
    ROOT / "services" / "misra_compliance.py",
    ROOT / "retrieval" / "retriever.py",
    ROOT / "ingestion" / "pdf_structure.py",
]
production_text = "\n".join(path.read_text(encoding="utf-8-sig") for path in production_files)
for idx, sentence in enumerate(rule106_questions + rule144_questions, 1):
    check(f"no full calibration-question hardcoding {idx}", sentence not in production_text)

failed = [item for item in checks if not item["pass"]]
summary = {
    "version": "v6.5.5.8",
    "validation": "PASS" if not failed else "FAIL",
    "checks": len(checks),
    "failures": len(failed),
    "failed_checks": failed,
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
raise SystemExit(0 if not failed else 1)
