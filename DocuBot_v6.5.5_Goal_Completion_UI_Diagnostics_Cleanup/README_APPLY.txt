DocuBot v6.5.5 - Goal Completion: UI + Diagnostics + Safe Final Cleanup
===============================================================================

BASE REQUIRED
-------------
Exact verified v6.5.4 only. The installer validates covered baseline hashes and
FINAL_COMPONENT_MANIFEST_v6.5.4.json before changing project files.

BEFORE APPLYING
---------------
1. Stop DocuBot / Streamlit completely, including its terminal/server process.
2. Do not run Update Knowledge Base at the same time.
3. Keep the current data/technical_documents and storage exactly as they are.

APPLY
-----
Extract this folder inside or next to the company-chatbot project, then run:

    Apply_v6.5.5_Goal_Completion_UI_Diagnostics_Cleanup.bat

The installer selects the project venv/.venv/env, verifies the patch and the
v6.5.4 baseline, backs up touched files, applies v6.5.5, runs static validation,
checks the existing semantic Rule index, runs read-only KB Health, and runs the
KB-update portability dry-run. It does NOT rebuild production Qdrant/BM25 and
does NOT run the live benchmark or live add/modify/delete lifecycle test.

V6.5.5 SCOPE
------------
- Fixes KB-maintenance UI stability: the maintenance area and chat/welcome
  composer are rendered before the long KB worker, so their normal positions
  remain present during an update.
- Confirmation checkbox remains available for both incremental and full rebuild
  modes; Update Knowledge Base stays disabled until explicitly confirmed.
- Preserves the already proven v6.5.4 lifecycle:
      unchanged -> SKIP
      new       -> process/add only
      modified  -> remove old + process/replace only
      deleted   -> remove only
- Preserves worker-only KB lock ownership, source-stability guard, and MISRA
  semantic profile scope.
- Adds a retrieval-only completion diagnostic for Recall, Precision, semantic
  consistency, MultiQuery decisions, and latency.
- Adds a separate LLM-from-verified-context diagnostic for grounding, reference
  correctness, Yes/No polarity, fallback behavior, and generation latency.
- Adds a completion suite that gates retrieval before LLM and checks the current
  normal-question combined latency target of 25 seconds.
- Adds an explicit safe live Add/Modify/Delete lifecycle probe. It is NOT run by
  the installer; it requires a clean noop baseline and includes recovery logic.
- Final cleanup archives exact known-obsolete manifests/validators/test BATs and
  old entry points under evidence/release_history. User-modified/hash-mismatched
  files are left untouched and reported instead of being deleted.
- Active production documentation is updated to the current Qwen2.5 + Qwen3
  embedding + BGE + Qdrant architecture. Historical evidence/logs are preserved.

PRESERVED INVARIANTS
--------------------
Generation model : qwen2.5:7b
Embedding model  : qwen3-embedding:8b
Reranker         : BAAI/bge-reranker-v2-m3
Chunking         : 900 / 150
Threshold        : 0.55
Top-K            : Vector 10 / BM25 10 / Final 3
MultiQuery       : original + exactly 2 alternatives
Hybrid retrieval : BM25 + Qdrant -> RRF -> BGE

AFTER A PASSING INSTALL
-----------------------
First send these two generated files for verification:

    logs\v6_5_5_completion\DocuBot_v6.5.5_Goal_Completion_UI_Diagnostics_Cleanup_Result_<timestamp>.zip
    logs\v6_5_5_completion\DocuBot_v6.5.5_Goal_Completion_UI_Diagnostics_Cleanup_Result_<timestamp>.zip.sha256.txt

Do not treat retrieval/LLM/latency goals as runtime-certified merely because the
installer passes. They are deliberately validated separately on the real PC.

After installation verification, the completion commands are:

    venv\Scripts\python.exe scripts\run_v6_5_5_completion_suite.py --skip-llm
    venv\Scripts\python.exe scripts\run_v6_5_5_completion_suite.py

Optional full lifecycle proof (adds/modifies/deletes a unique temporary TXT and
returns the KB to noop):

    venv\Scripts\python.exe scripts\run_v6_5_5_completion_suite.py --with-live-kb-lifecycle

Or run only the live lifecycle probe:

    venv\Scripts\python.exe scripts\test_incremental_kb_lifecycle_live.py

The UI check remains visual/manual: during Update Knowledge Base, the #1
welcome/chat composer and #2 maintenance controls should stay in their normal
positions, with the confirmation checkbox still visible and the Update button
disabled while the worker is active.
