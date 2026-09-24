DocuBot v6.5.5.4 — KB Notification Card Readability Hotfix
===========================================================

Purpose
-------
The v6.5.5.3 native Streamlit success toast still clips the second line on the
observed browser/Streamlit combination. This hotfix replaces ONLY the successful
KB-update toast with a DocuBot-owned fixed notification card whose full title and
Added/Updated/Deleted counts are under our CSS control.

Preserved
---------
- Chat textarea/send remain disabled while a KB update is running.
- #1 welcome/chat region and #2 maintenance region stay in their normal positions.
- Incremental KB lifecycle remains unchanged: unchanged=SKIP; new/modified/deleted=PROCESS.
- Qwen2.5 generation, Qwen3 embeddings, BGE reranker, Qdrant + BM25 preserved.
- Chunking 900/150, retrieval score 0.55, Top-K 10/10/3, exactly 2 MultiQuery alternatives preserved.
- Installer does NOT rebuild the production KB.

Apply
-----
1. Stop DocuBot/Streamlit completely.
2. Extract this ZIP inside the project folder (or keep the patch folder directly under it).
3. Run: Apply_v6.5.5.4_KB_Notification_Card_Readability_Hotfix.bat
4. Target: PASS.
5. Restart DocuBot and make one harmless TXT modification/update.

Expected notification
---------------------
Knowledge Base is successfully updated.
Added: X   Updated: Y   Deleted: Z

All three counts must be fully readable. Technical details remain in logs only.
