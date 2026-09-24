DocuBot v6.5.5.8 — Natural Query Calibration / Generalization
==============================================================

BASELINE
- Built from the cleaned current-working project supplied by the user.
- This is a narrow calibration patch based on actual manual QA evidence.

WHAT IT FIXES (generic, source-grounded)
1. "externally visible" C-linkage wording is no longer mistaken for a request to use external knowledge.
2. Deterministic MISRA source-language cues are checked against the authoritative corpus before expensive semantic MultiQuery when they are strong enough.
3. Natural descriptions of Rule 8.7 / Rule 10.6 / Rule 14.4 concepts are grounded through requirement wording, not phrase->Rule hardcoding.
4. Prose like "if(counter)" or "empty else block" no longer falsely injects the Rule 15.6 unbraced-body cue.
5. Generic empty-else wording stays conditional and uses Rule 15.7 context correctly.
6. Structured list/family results are protected from collapsing into one Rule during post-retrieval promotion.
7. Structured catalog stopwords are stem-normalized, preventing "related" -> "relat" from pulling unrelated Rules.
8. A real parser boundary defect at Rule 22.2 is fixed. Runtime exact lookup/category inventory can recover the full source block without a forced re-embedding.

UNCHANGED
- qwen2.5:7b generation model
- qwen3-embedding:8b embedding model
- BGE reranker
- chunk size / overlap: 900 / 150
- global retrieval threshold: 0.55
- Top-K settings
- Qdrant vectors / active KB storage
- KB updater transaction logic

IMPORTANT
This installer DOES NOT rebuild or re-embed the MISRA PDF. The parser correction will be used on a future legitimate re-ingestion. For the current active KB, runtime source recovery covers the one known split Rule 22.2 record.

HOW TO RUN
1. Close DocuBot / Streamlit.
2. Put this whole folder directly under C:\user_dev\company-chatbot\
3. Double-click Run_v6.5.5.8_Natural_Query_Calibration.bat
4. If validation passes, open DocuBot normally and run CALIBRATION_SET_A_CRITICAL_RETEST.txt.
5. Send the newest Result ZIP + SHA256 from logs\v6_5_5_8_calibration back for review.

ROLLBACK
If any apply/compile/static validator fails, the runner restores the previous files automatically.
