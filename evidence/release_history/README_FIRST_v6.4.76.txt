DocuBot v6.4.76 — Phase 7 Certification Harness Hotfix
=======================================================

BASELINE REQUIRED
-----------------
Apply this patch over v6.4.75.

WHY THIS EXISTS
---------------
v6.4.75 correctly passed production-v4 preflight and the OOD safety probe on
the lower-spec personal PC, but the Architecture Lock stopped because old
release-note TXT files were absent. The early exit also happened before the
combined diagnostic ZIP was created.

WHAT v6.4.76 CHANGES
--------------------
1. V6_4_*_NOTES.txt files are optional audit evidence, not hard runtime-lock
   dependencies. Missing notes warn but do not pass over a real production
   code/config/hash mismatch.
2. The combined certification ZIP is created even when an early gate fails.

RUN
---
Run_Phase7_Optimization1_Full_Certification.bat

UPLOAD
------
logs\phase7_optimization1_certification\
DocuBot_Phase7_Optimization1_Full_Certification_Result_*.zip

NO KB REBUILD / NO PIP INSTALL.
