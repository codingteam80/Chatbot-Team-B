DocuBot v6.5.5.1 - KB Update Notification UX Hotfix
======================================================================

PURPOSE
-------
Simplify only the successful Update Knowledge Base notification.

Expected user-facing toast:
  Knowledge Base is successfully updated.
  Added: N · Updated: N · Deleted: N

Technical details (chunks, unchanged/skipped, Qdrant/BM25, semantic Rule index,
KB Health) remain in logs/evidence and are intentionally hidden from the normal
success notification.

PRESERVED
---------
- v6.5.5 stable KB-update layout and confirmation checkbox
- unchanged=SKIP; new/modified/deleted=PROCESS incremental lifecycle
- worker lock and source-stability guard
- Qwen2.5:7b generation
- qwen3-embedding:8b embeddings
- BAAI/bge-reranker-v2-m3 reranker
- chunking 900/150
- retrieval threshold 0.55
- Top-K 10/10/3
- original + exactly 2 MultiQuery alternatives
- retrieval/LLM/latency completion diagnostics

INSTALL
-------
1. Stop DocuBot/Streamlit completely.
2. Extract this ZIP.
3. Run Apply_v6.5.5.1_KB_Update_Notification_UX_Hotfix.bat.
4. Target: PASS and a generated Result ZIP + SHA256.
5. Restart DocuBot and perform one harmless KB update to visually verify the toast.

The installer does NOT rebuild the production KB.
It verifies the exact v6.5.5 baseline, creates a rollback backup, validates the
new UI helper, runs read-only KB Health, and restores the previous files if a
post-copy check fails.
