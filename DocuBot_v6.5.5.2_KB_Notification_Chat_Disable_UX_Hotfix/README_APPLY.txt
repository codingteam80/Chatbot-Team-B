DocuBot v6.5.5.2 - KB Notification Polish + Chat Disable UX Hotfix
===================================================================

WHAT THIS CHANGES
-----------------
1. Makes the successful Update Knowledge Base notification cleaner/readable:
   Knowledge Base is successfully updated.
   Added: N · Updated: N · Deleted: N
2. Keeps technical chunk/Qdrant/BM25/semantic-index/KB-health details in logs only.
3. Disables the chat textarea AND send button while a KB update is running.
4. Adds a server-side chat submit guard during the KB update.
5. Keeps the chat/welcome layout in the same position while disabled.

WHAT THIS DOES NOT CHANGE
-------------------------
- Incremental KB lifecycle
- Qdrant/BM25 data
- Retrieval / MultiQuery / RRF / reranker behavior
- Generation model / embedding model / reranker model
- Chunking 900/150
- Threshold 0.55
- Top-K 10/10/3

HOW TO APPLY
------------
1. Stop DocuBot/Streamlit completely.
2. Extract this ZIP into the DocuBot project root so this patch folder is directly inside it.
3. Run Apply_v6.5.5.2_KB_Notification_Chat_Disable_UX_Hotfix.bat.
4. Target: PASS. Installer performs read-only KB Health only; it does NOT rebuild the KB.
5. Restart DocuBot and run one harmless Update Knowledge Base test.

The installer verifies exact v6.5.5.1 code hashes before changing files and automatically restores the previous files if validation or KB Health fails.
