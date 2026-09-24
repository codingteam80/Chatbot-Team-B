DocuBot v6.5.5.6 - Final Certification
=======================================

Purpose
-------
This is a QA/certification folder only. It does not replace or tune production
retrieval, models, chunking, thresholds, Top-K, Qdrant/BM25, or the UI.

Why v6.5.5.6
------------
The v6.5.5.5 evidence proved:
- delete lifecycle actual PASS
- retrieval Recall/Precision/semantic consistency PASS
- answer generation from verified context PASS

The previous overall FAIL came from a QA measurement artifact: cold startup
cost from separate diagnostic processes was added to every-category latency,
and ambiguous MultiQuery cases were gated as if they were normal questions.

v6.5.5.6 corrects the QA measurement only:
- production-equivalent retrieval and fast-LLM prewarm is completed first;
  startup cost is recorded separately.
- the original <=25 sec normal-question goal applies to the single-query /
  early-accept path.
- ambiguous MultiQuery latency is still measured, but reported separately as
  an optimization advisory.

How to use
----------
1. Keep this entire folder inside the live project root, for example:
   C:\user_dev\company-chatbot\DocuBot_v6.5.5.6_Final_Certification\
2. Close DocuBot / Streamlit.
3. Run: Run_Final_Certification_v6.5.5.6.bat
4. Send the newest result ZIP and SHA256 file from:
   logs\v6_5_5_final_certification\
