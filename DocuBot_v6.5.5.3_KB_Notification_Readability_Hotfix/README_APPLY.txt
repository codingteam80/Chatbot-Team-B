DocuBot v6.5.5.3 — KB Notification Readability Hotfix
=====================================================

Scope
-----
This is a focused UI-only hotfix on top of exact v6.5.5.2.

Fix
---
- Keeps the same top-right success notification.
- Ensures the full title is readable.
- Ensures Added / Updated / Deleted counts are fully visible.
- Removes fixed-height/overflow clipping from the Streamlit toast.
- Keeps responsive width on smaller screens.

Preserved
---------
- Chat remains disabled while Update Knowledge Base is running.
- UI positions remain stable.
- Incremental KB lifecycle is unchanged.
- Unchanged files remain skipped.
- Retrieval, MultiQuery, LLM, models, chunking 900/150, threshold 0.55,
  and Top-K 10/10/3 are unchanged.
- Installer performs no Qdrant/BM25 rebuild.

Apply
-----
1. Stop DocuBot / Streamlit completely.
2. Extract this ZIP beside the company-chatbot project folder.
3. Run Apply_v6.5.5.3_KB_Notification_Readability_Hotfix.bat.
4. Target: PASS.
5. Restart DocuBot.
6. Modify the harmless Employee_Leave_Test.txt and run one KB update.
7. Verify the entire success toast is visible.
