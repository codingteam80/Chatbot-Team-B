DocuBot v6.5.5.7 - Final Answer Closure + Chunking Evidence
===========================================================

PURPOSE
-------
Close only the last answer-generation quality gap proven by the v6.5.5.6 evidence:
- preserve the sole verified Rule/Directive reference in the final answer;
- prevent unrelated Unicode-script drift after verified retrieval.

This change is generic and corpus-driven. It does NOT contain phrase-to-Rule mappings.
The Rule/Directive identifier can come only from accepted retrieval-result metadata.

UNCHANGED
---------
- qwen3-embedding:8b
- qwen2.5:7b
- BAAI/bge-reranker-v2-m3
- chunk size 900
- chunk overlap 150
- Top-K values
- retrieval threshold
- MultiQuery routing
- incremental KB updater
- Qdrant/BM25 transaction logic

CHUNKING EVIDENCE
-----------------
The runner also generates a read-only evidence ZIP containing:
- active chunk configuration and QA checks
- one-row-per-chunk inventory CSV
- full text of every active chunk
- focused chunks for Rules/Directives used by the latest retrieval diagnostic
- active manifest snapshot
- optional read-only Qdrant point-count cross-check

It does NOT rebuild, re-embed, modify, or delete KB data.

HOW TO RUN
----------
1. Copy this whole folder directly under:
   C:\user_dev\company-chatbot
2. Close DocuBot / Streamlit.
3. Run:
   Run_v6.5.5.7_Final_Answer_Closure.bat
4. Send the newest Result ZIP and SHA256 from:
   C:\user_dev\company-chatbot\logs\v6_5_5_7_closure

If any closure stage fails, the runner automatically restores the prior production answer-service file.
