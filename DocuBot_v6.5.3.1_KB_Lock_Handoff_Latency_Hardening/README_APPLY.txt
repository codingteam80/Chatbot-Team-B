DocuBot v6.5.3.1 - KB Update Worker Lock Handoff + Latency Decision Diagnostics
================================================================================

WHAT THIS FIXES
---------------
1) FIELD-OBSERVED UPDATE KNOWLEDGE BASE SELF-LOCK
   v6.5.3 could show:
     Another DocuBot knowledge-base update is already running.
     Lock owner: PID <Streamlit PID> from streamlit_button.

   v6.5.3.1 makes the isolated worker the ONLY owner of the cross-process
   knowledge-base update lock. The launcher performs a safe handoff check but
   never owns the worker lock itself.

2) STALE/PID-REUSE LOCK HARDENING
   - dead locks are cleaned safely;
   - new worker locks record run_id, parent_pid, owner_role and process-start
     identity;
   - a real live worker lock still blocks any second update;
   - the exact legacy v6.5.3 Streamlit parent self-lock can be retired safely.

3) STREAMLIT DOUBLE-CLICK GUARD
   The browser session marks an update as in progress while the isolated worker
   is running, preventing a repeated button launch in the same session.

4) LATENCY/MULTIQUERY TRACE HARDENING
   v6.5.3 single-query early-accept and prewarm logic are preserved. No semantic
   acceptance threshold is changed in this hotfix. The runtime now records the
   exact timing/decision split for:
     - resolver readiness,
     - original single-query resolution,
     - MultiQuery generation,
     - widened semantic resolution,
     - total Rule-resolution time,
     - whether MultiQuery was skipped or actually needed.

PRESERVED
---------
- Generation model: qwen2.5:7b
- Embedding model: qwen3-embedding:8b
- Reranker: BAAI/bge-reranker-v2-m3
- Vector/BM25/Final Top-K: 10/10/3
- Minimum retrieval score: 0.55
- Chunk size/overlap: 900/150
- MultiQuery: original + exactly 2 alternatives, RRF
- Existing Qdrant/BM25 knowledge base
- v6.5.3 Yes/No tuple fix
- v6.5.3 self-contained follow-up isolation
- v6.5.3 single-query semantic+BGE early accept

IMPORTANT BEFORE APPLYING
-------------------------
1. CLOSE DocuBot / Streamlit first.
2. Do not manually delete Qdrant/BM25 storage.
3. Extract this ZIP next to your company-chatbot project, same way as prior
   exact patches.
4. Run:
     Apply_v6.5.3.1_KB_Lock_Handoff_Latency_Hardening.bat

The installer is fail-closed:
- exact v6.5.3 baseline hashes are required;
- a live KB-update lock blocks installation before any project file is changed;
- stale dead lock can be removed safely;
- payload hashes are verified;
- backup is created;
- v6.5.3.1 validator must PASS;
- v6.5.3 regression validator must PASS;
- semantic Rule source/index readiness must PASS;
- KB Health must PASS;
- KB update portability/lock-handoff dry-run must PASS;
- otherwise code is rolled back to v6.5.3.

AFTER INSTALLER PASS
--------------------
1. Start DocuBot locally.
2. If the warning says KB maintenance is needed, tick the confirmation.
3. Click Update Knowledge Base ONCE.
4. Expected flow:
     launcher handoff -> worker owns lock -> preflight -> transactional build ->
     semantic Rule index verification -> KB Health -> PASS -> lock released.
5. If the update still fails, send the NEW files shown by the UI:
     logs\kb_update\kb_update_result_*.json
     logs\kb_update\kb_update_console_*.txt
6. Do not run the mixed benchmark until the KB button end-to-end test is clean.
