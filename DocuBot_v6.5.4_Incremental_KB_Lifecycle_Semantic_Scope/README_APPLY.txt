DocuBot v6.5.4 - Incremental KB Lifecycle + Semantic Scope + Source Stability
================================================================================

WHAT THIS PATCH FIXES
---------------------
1. Restores the intended Update Knowledge Base lifecycle:
   - UNCHANGED -> SKIP existing vectors, no re-embedding
   - NEW       -> parse/chunk/embed/add only the new file
   - MODIFIED  -> remove old chunks then parse/chunk/embed/replace only that file
   - DELETED   -> remove only that file's vectors/manifest entry
   - FULL REBUILD only for incompatible/missing/corrupt index identity

2. Keeps the v6.5.3.1 worker-only lock handoff and double-click protection.

3. Adds source-change safety:
   - changed/new files are copied to a private transaction snapshot
   - source hashes are checked again before commit
   - the full-rebuild path also refuses commit if source files changed mid-run

4. Fixes unnecessary derived MISRA semantic-index work:
   - readiness now depends on extracted Rule/Directive profile content
   - unrelated office document changes do NOT re-embed all 173 MISRA profiles
   - compatible v1 semantic metadata migrates without model inference

5. Preserves the latency/MultiQuery hardening already added:
   - strong single-query semantic+BGE agreement can skip MultiQuery
   - ambiguous queries still use original + exactly 2 alternatives
   - semantic latency decision diagnostics remain enabled

PRESERVED
---------
- Generation model: qwen2.5:7b
- Embedding model: qwen3-embedding:8b
- Reranker: BAAI/bge-reranker-v2-m3
- Chunk size/overlap: 900/150
- Minimum retrieval score: 0.55
- Vector/BM25/Final Top-K: 10/10/3
- No production Qdrant/BM25 rebuild during patch installation

BASELINE
--------
Installer accepts the exact verified v6.5.3 or v6.5.3.1 code state for the
covered files. This is intentional because field evidence showed the latest KB
result still reported v6.5.3 even after the lock-handoff work was discussed.
Unknown or modified baselines are blocked before files are changed.

HOW TO APPLY
------------
1. Stop DocuBot / Streamlit completely. Do not leave the server console running.
2. Extract this ZIP beside the company-chatbot project.
3. Run:
      Apply_v6.5.4_Incremental_KB_Lifecycle_Semantic_Scope.bat
4. Target result: PASS.
5. Send the generated v6.5.4 Result ZIP + .sha256.txt for verification.

IMPORTANT
---------
The installer validates code, KB Health, portability, lock behavior, Qdrant
staging/delta APIs, and performs only a metadata-only semantic-index signature
migration when compatible. It does NOT click Update Knowledge Base and does NOT
rebuild production Qdrant/BM25.

FIELD TEST AFTER INSTALL
------------------------
Use a small test change and click Update Knowledge Base once.
Expected if MISRA_FromInternet.pdf is unchanged and one other file is new:

  Mode             : incremental
  Added            : 1
  Unchanged/skip   : 1
  Action           : transactional incremental Qdrant update
  [ADDED] <file> (... chunks)
  Unchanged skipped       : 1
  Changed chunks embedded : <only the new file chunks>

There must NOT be a full 951-chunk MISRA re-embedding pass.
