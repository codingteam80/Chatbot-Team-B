DocuBot v6.4.75 — Phase 7 Optimization #1 Full Certification
=============================================================

BASELINE REQUIRED
-----------------
Apply this exact patch over v6.4.74 Phase-7 Optimization #1 Trial.
The promoted production Chunking-v4 storage from v6.4.72 must already be ready.

SCOPE
-----
Certification/tooling only. No production answer/retrieval optimization is changed.
This release combines four acceptance gates in one run:
  - Optimization #1: 7/7 quality + 7/7 activation guards
  - Technical QA: 16/16
  - English Natural: 19/19
  - Active-corpus identity-OOD safety probe

RUN
---
Run_Phase7_Optimization1_Full_Certification.bat

UPLOAD
------
logs\phase7_optimization1_certification\
DocuBot_Phase7_Optimization1_Full_Certification_Result_*.zip

NO KB REBUILD / NO PIP INSTALL.
