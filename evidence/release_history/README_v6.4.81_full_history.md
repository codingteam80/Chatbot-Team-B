## v6.4.72 — Windows-safe Chunking-v4 Production Promotion

v6.4.72 is a narrow office-promotion hotfix for the actual Windows `WinError 5` seen when v6.4.71 tried to rename the fully validated staging directory into `storage/option_c_qwen3_qdrant_v4`. The certified v4 content itself remains unchanged. Promotion now builds and validates staging exactly as before, then copies that validated tree into a fresh final destination instead of renaming/replacing a directory. A final `metadata/v4_production_ready.json` marker is written only after destination validation, so a partial copy is never selected as the normal v4 read path.

The utility can automatically repair an uncertified partial v4 destination left by a prior failed promotion, but it never removes the certified v3 rollback or Experimental B source. If Windows still has the partial production-v4 directory open, promotion fails with an explicit close-process message rather than touching protected stores. No model, retrieval, chunking, index contents, source documents, thresholds, or answer behavior change.

---

## v6.4.71 — Phase-6 Chunking-v4 Production Promotion Baseline

v6.4.71 closes roadmap Phase 6 by promoting the already-certified Chunking-v4 representation to the normal production default. The v4 chunking algorithm, `CHUNK_SIZE=900`, `CHUNK_OVERLAP=150`, Qwen2.5:7B generation, Qwen3-Embedding-8B, BGE reranker, retrieval thresholds, prompts, answer logic, and source documents are unchanged from the v6.4.70 certification candidate.

The certified Experimental B index is **copied and verified**, not renamed, into `storage/option_c_qwen3_qdrant_v4`. The Phase-5/v3 index at `storage/option_c_qwen3_qdrant` remains untouched as an explicit rollback. Normal settings now default to `DOCUBOT_CHUNKING_PROFILE=v4` and the production v4 storage root; `DOCUBOT_CHUNKING_PROFILE=v3` selects the retained rollback path.

On an office PC that already has the certified Experimental B from v6.4.68-v6.4.70, apply the patch and run `Promote_Chunking_v4_To_Production.bat` once. The promotion utility verifies v4 schema/embedding metadata, clones Qdrant SQLite through SQLite backup, compares logical table counts, excludes `.lock`/WAL/SHM runtime files, verifies the destination, and never deletes the v3 or Experimental B stores. Use `Start_DocuBot_v4_Production.bat` for an explicit production launch or `Start_DocuBot_v3_Rollback.bat` for immediate Phase-5 rollback.

Promotion evidence: v6.4.70 Experimental B full certification on office Windows/Ollama passed Technical **16/16** and English Natural **19/19** with the final EN-011/FU-T2 manual cleanup verified. This release changes selection/storage promotion only; it does not introduce a new chunking experiment.

---

## v6.4.70 — Phase-6 Chunking-v4 Promotion Cleanup

v6.4.70 keeps the exact v4 index/chunking design from v6.4.69 and fixes two manual-review artifacts before promotion: terminal parent Rule/Directive anchors are removed from visible Example/Rationale text, and exact-structured MISRA rationale follow-ups are finalized directly from the retrieved source Rationale instead of using a generic LLM rewrite. No index rebuild, model, retrieval, threshold, chunk-size, overlap, document, or storage change is introduced.

Run `Run_Phase6_Chunking_v4_Full_Certification.bat` against the existing Experimental B index. Promotion requires Technical 16/16 and English Natural 19/19.

---

## v6.4.69 — Phase-6 Chunking-v4 Full Certification Trial

v6.4.69 does **not** change the v4 chunker. It adds an isolated certification wrapper that refuses to run unless `storage/experiments/chunking_v4` contains a genuine v4/Qwen3 index, then runs the existing Technical 16/16 and English 19/19 suites with process-local v4 storage selection. Control A (`storage/option_c_qwen3_qdrant`) remains the default and is never promoted or overwritten automatically. A single combined ZIP is written under `logs/chunking_v4_certification`.

Run `Run_Phase6_Chunking_v4_Full_Certification.bat` after the v4 Experimental B index has been built. A PASS only makes v4 a promotion candidate; latency comparison and manual semantic review remain required before any production promotion.

## v6.4.68 — Phase-6 Chunking-v4 Experimental A/B Trial

v6.4.68 starts roadmap Phase 6 without replacing the certified v6.4.67 index. **Control A remains the current v3 structure-aware index at `storage/option_c_qwen3_qdrant`.** The new v4 profile is opt-in only (`DOCUBOT_CHUNKING_PROFILE=v4`) and builds to the separate `storage/experiments/chunking_v4` root. Default DocuBot startup, the production Option-C builder, models, prompts, retrieval thresholds, Top-K values, answer logic, documents, and certified v3 storage remain unchanged.

Chunking-v4 refines the already structure-aware PDF path rather than replacing it. Subordinate MISRA sections such as `Rationale`, `Example`, `Amplification`, and `Exception` inherit their parent Rule/Directive anchor, while inline `See also` tails are separated into dedicated cross-reference units. This preserves all source text but reduces cross-reference token mixing inside substantive evidence. The existing 900/150 size/overlap limits remain unchanged.

The trial adds `Build_DocuBot_Chunking_v4_Experimental_Index.bat`, `Run_Chunking_v4_AB_Benchmark.bat`, and the one-command `Run_Phase6_Chunking_v4_Trial.bat`. The A/B benchmark snapshots both Qdrant arms read-only, runs the same retrieval pipeline against 10 role-sensitive natural questions, and treats any Experimental-B evidence-coverage regression as a rejection. A promising A/B result does **not** promote v4 automatically; it only unlocks full Technical 16/16 + English 19/19 certification against Experimental B. Rollback is immediate because Control A is never overwritten.

The frozen Phase-5 baseline is v6.4.67: office Technical **16/16 PASS** plus the already accepted English Natural **19/19 PASS**.

## v6.4.67 — Phase-5 Technical QA Qdrant isolation

v6.4.67 is a certification-harness-only follow-up to the actual v6.4.66 office rerun. v6.4.66 proved the 16-case suite was already running in one shared Python/AnswerService lifecycle, but OC-011 still hit the production embedded-Qdrant `qdrant/.lock`. That result isolates the remaining contention outside the case-worker architecture: another live DocuBot/runtime can own the production Qdrant directory while certification reaches its first full vector-search path.

The Technical QA harness now creates a **run-specific isolated Qdrant snapshot before `AnswerService` starts**. It never opens a Qdrant client against the production directory, never copies/deletes/modifies production `qdrant/.lock`, and excludes SQLite WAL/SHM/journal runtime sidecars. Qdrant SQLite payloads are cloned with SQLite's read-only backup API so the QA copy is transactionally consistent even when the production database is open. The 16 cases then run in one shared process against only that isolated snapshot, with fresh chat/session state per case. The temporary snapshot is removed on best effort after the result ZIP is written.

No product answer logic, model, embedding, reranker, retrieval threshold, Top-K, chunking/index schema, technical documents, production Qdrant/BM25 storage, or KB contents are changed. English Natural **19/19 PASS** from v6.4.65 remains accepted; only the Option-C Technical **16/16** office rerun is required for the final Phase-5 gate.

## v6.4.66 — Phase-5 technical QA harness architecture fix

v6.4.66 is a harness-only follow-up to the actual v6.4.65 office reruns. v6.4.65 English Natural reached **19/19 PASS**, while two consecutive Option-C Technical runs remained **15/16** because OC-011 alone hit the same Windows embedded-Qdrant `qdrant/.lock` PermissionError. The bounded retry ran three times, but the following independent case process passed, isolating the remaining problem to certification-process churn rather than a new DocuBot semantic or retrieval defect.

The normal Technical QA run now executes all 16 cases in **one Python process with one shared `AnswerService` / retriever lifecycle**, while resetting `ChatManager` to a completely fresh conversation before every case. This keeps test cases independent without repeatedly tearing down and reopening the embedded Qdrant runtime across separate worker processes. Per-case JSON/console evidence and the result ZIP are preserved. The harness never deletes or modifies the Qdrant `.lock` file.

No answer logic, model, embedding, reranker, retrieval threshold, Top-K, chunking/index schema, technical documents, KB, or active Qdrant/BM25 storage is changed. English Natural does not need another rerun for this harness-only patch; the remaining Phase-5 gate is Option-C Technical **16/16** on office Windows/Ollama.

## v6.4.65 — Phase-5 certification cleanup

v6.4.65 is a narrow certification/harness cleanup built from the actual v6.4.64 office rerun. The EN-007 final answer was semantically correct, but the benchmark re-validator treated the word `satisfied` inside uncertainty prose (`does not establish whether this requirement is satisfied`) as an affirmative compliance claim. The Option-C Technical suite also hit one transient Windows embedded-Qdrant `.lock` PermissionError in OC-011.

This release makes semantic-status detection polarity-aware for uncertainty-qualified wording while still rejecting real `Status: Satisfied` promotion. The Technical QA harness adds a bounded retry only for the exact embedded-Qdrant `.lock` PermissionError; it never deletes or modifies the lock file, and persistent contention still fails. No model, retrieval, index, KB, document, threshold, Top-K, chunking, or active-storage change is included.

Office acceptance remains Option-C Technical **16/16** plus English Natural **19/19**. No KB rebuild or `pip install` is required.

---

# DocuBot

## v6.4.64 — Phase-5 semantic claim fidelity closure candidate

v6.4.64 is a narrow follow-up to the actual v6.4.63 office rerun. v6.4.63 reached **16/16 Option-C Technical QA** and **19/19 English Natural automated PASS**, but manual review of EN-007 found that one generated switch review still contained incorrect visible-code claims while the automated semantic guard reported PASS. The defect was in per-Rule answer segmentation, not retrieval, model selection, or source grounding: Markdown list prefixes such as `- **Rule 16.2**` prevented the guard from isolating each Rule block, so later Rule 16.7 uncertainty text could mask earlier uncertain-to-satisfied promotions.

v6.4.64 replaces that newline-format assumption with exact Rule/Directive reference segmentation, explicitly rejects **uncertain → satisfied** promotion, mechanically extracts current switch facts (visible expression, case/default label counts, and default position), carries those facts into the grounded generation contract, and rejects generated claims that change them. For the EN-007 snippet, the contract now records that `x` is visible but its essential type is unknown, there is **one case label**, one default label, and the default is **last**. The benchmark re-runs the production semantic alignment guard for code-review cases and explicitly forbids the old false-pass wording.

No LLM/model, embedding, reranker, retrieval threshold, Top-K, chunking/index schema, company/technical document, Qdrant/BM25 storage, or KB rebuild behavior changes. Local v6.4.60-v6.4.64 focused tests are **42/42 PASS**. The retained v6.4.39-v6.4.64 block is **203 PASS / 2 baseline FAIL**; the same two v6.4.53/v6.4.54 recovery-marker assertions predate this patch and were already present in v6.4.63. Office Windows/Ollama Technical 16/16 + English Natural 19/19 rerun remains the final Phase-5 exit gate.

## v6.4.63 — Phase-5 semantic quality closure candidate

v6.4.63 is a narrow corrective release built from the actual v6.4.62 office project after the Phase-5 English Natural Benchmark reached 17/19 automated PASS and manual review exposed several false-positive semantic passes. It keeps the Option-C Qwen2.5/Qwen3-Embedding/Qdrant/BGE architecture, thresholds, chunking, documents, and storage unchanged.

The patch stops compact Example/Rationale extraction at the next inline source role (including `See also`), retains the complete three-sentence Rule 16.7 rationale when the concluding `if-else` recommendation is material, treats explicit follow-up facts such as “the default label is first” as established facts, preserves the exact visible pointer operator, and attaches conservative deterministic states to complete structured Rule-family evidence before LLM synthesis. The generation contract and semantic alignment guard now reject a model that claims an unknown switch-expression type is satisfied or rewrites `p = p + 2` as `p += 2`. The Phase-5 and Option-C validators were strengthened so those former false passes cannot silently pass again.

Local focused validation: v6.4.63 + retained v6.4.62 semantic tests pass. The broader v6.4.39-v6.4.63 retained block shows only the same two pre-existing v6.4.53/v6.4.54 source-recovery marker assertions that also fail on an untouched v6.4.62 extract; no new failure was introduced by this patch. Office Windows/Ollama rerun remains required for final Phase-5 closure.

## v6.4.60 — English-natural quality + conservative cleanup

v6.4.60 preserves the v6.4.58/v6.4.59 Option-C model/index architecture while improving English-first natural-user handling: intent-aware exact Rule explanations and comparisons, structured multi-Rule retrieval, safer follow-up/topic reset behavior, context hygiene that keeps authoritative Rule bodies ahead of rationale/examples, broader multi-Rule code-review support when explicitly requested, and dynamic composer safe-area/scroll anchoring. Tagalog/Taglish remains supported as a secondary normalization path.

This release also removes four zero-reference prototype modules and one stale packaged A/B-result artifact. Legacy Chroma rollback storage, active Qdrant/BM25 data, documents, tests, release notes, and historical runtime logs are intentionally retained. The cleanup is conservative and does not change models, thresholds, chunking, corpus, cache schema, or index schema. Rule-aware chunking v4 remains a separate future A/B experiment rather than an in-place migration.

## v6.4.59 — Dynamic MISRA guidance + ChatGPT-like response UX

Office v6.4.58 Option-C Technical QA reached **16/16 PASS** on 2026-09-17, closing the Rule-family completeness/cap issue. Manual UI testing then exposed two separate quality gaps: prospective questions such as planning an `if-else` to `switch` refactor were being misclassified as current compliance violations, and assistant/loading/source containers still behaved like fixed cards with large blank areas. v6.4.59 keeps the certified v6.4.58 retrieval/model/index architecture, adds intent-based prospective MISRA guidance that does not invent a violation verdict without code/evidence, aggregates same-file source traces, and renders assistant replies/loading/sources as an unboxed content-driven conversational flow.

The sample manual questions are regression examples only; production routing is intent/pattern based rather than exact-question or canned-answer matching. No model, embedding, reranker, threshold, active corpus, cache schema, or index schema change.

## v6.4.58 — Structured-family context-cap fix

Office v6.4.57 QA proved the structured-family route was active but the final answer still contained only six family members. Root cause: `CompanyRetriever.build_context()` reapplied the generic context Top-K after `retrieve()` had already assembled the complete source-grounded structured family. v6.4.58 preserves complete structured-family inventories through context assembly while leaving the normal Top-K cap unchanged for ordinary retrieval. No model, embedding, reranker, source-data, cache-schema, or index-schema change.

## v6.4.57 — Structured-family pre-rescue routing fix

Office diagnostics proved Rule 16.7 was present in the active BM25 corpus, raw PDF, structure parser, prepared cache, and structured-family reconciliation. The remaining 15/16 failure was caused by routing order in `AnswerService`: the older MISRA rule-body cue rescue (bounded to Top-K 6 for assessment questions) ran before the complete structured-family path, truncating a seven-member family to six. v6.4.57 lets source-grounded structured-family results bypass that bounded rescue. Unsupported list topics continue through the existing rescue/hybrid path. No model, embedding, reranker, threshold, source-data, cache-schema, or index-schema change.

# v6.4.50 — Option C Technical/MISRA Trial

## v6.4.56 — Authoritative-source path resolution

Actual office v6.4.55 QA remained **15/16 PASS** with only OC-014 missing
Rule 16.7; OC-015 and OC-016 remained fixed. The returned OC-014 evidence
contained only the live BM25 Rule 16.1–16.6 chunks and no source-recovery
marker. v6.4.56 removes the remaining dependency on the historical absolute
source path stored in index metadata: structured family reconciliation now
resolves the authoritative PDF from the active `data/technical_documents`
corpus by source filename and scans that source directly. The Rule-family major
is derived from the retrieved structured records and only explicit source-backed
Rule entries are admitted. No Rule 16.7 requirement text, model change,
embedding change, reranker change, threshold change, or index rebuild is
hardcoded into this correction.

## v6.4.55 — Prepared-source family recovery

Actual office v6.4.54 QA again reached **15/16 PASS**. OC-015 and OC-016
remain fixed; only OC-014 still omitted Rule 16.7. Inspection of the same
technical source showed that the structure-aware prepared-ingestion artifact
contains an explicit Rule 16.7 chunk even though the live BM25 family records
do not expose it. v6.4.55 therefore adds a layered source-backed recovery path:
active structured records first, then the prepared ingestion cache, then the
existing structured source-PDF parser, then a conservative direct PDF-text
scan, and finally the indexed Appendix A summary. Only explicit Rule-family
entries with source text are admitted; no Rule 16.7 requirement or canned
answer is embedded in production logic. Model, embedding, reranker, thresholds,
chunking, active corpus, and index schema remain unchanged.

## v6.4.54 — Indexed-summary family recovery

Actual office v6.4.53 QA again reached **15/16 PASS**. OC-015 (Rule 16.6
minimum-count application) and OC-016 (trigraph polarity) remained fixed; the
only failure was OC-014 because the live source-PDF supplement added no
`_structured_source_recovered` member on the office Windows runtime. The active
indexed corpus already contains the authoritative `Appendix A: Summary of
guidelines` entry with the explicit Rule 16.1–16.7 sequence. v6.4.54 adds a
bounded fallback that parses only literal Rule headings from that indexed
summary when the direct Rule-family records are incomplete. No requirement text
is canned and no model, embedding, reranker, threshold, chunking, or index
schema changes.

## v6.4.53 — Option C source-family reconciliation

Actual office v6.4.52 QA reached **15/16 PASS**. The remaining failure was a
structured-list completeness gap: the active index supplied Rule 16.1–16.6,
while the authoritative technical PDF still contained Rule 16.7. v6.4.53 keeps
the Option-C model/retrieval architecture unchanged and adds a bounded cached
source-PDF reconciliation step for an already-proven structured Rule family.
Only explicit Rule headings from the same source PDF can supplement missing
indexed family members; no answer text is hardcoded.


Built strictly from the user-supplied v6.4.49 checkpoint. This is a **trial / pending Windows-Ollama certification** checkpoint focused on MISRA and future technical/technology documents rather than the earlier history/general-corpus test material. The v6.4.49 legacy Chroma/E5/llama+qwen stack remains available through `Start_DocuBot_Certified_Default.bat` as an explicit rollback path.

## Option C production candidate

- Active default knowledge profile: `technical`.
- Active source folder: `data/technical_documents`; the bundled trial corpus contains only the byte-identical `MISRA_FromInternet.pdf`. History/Wikipedia/Leave Policy test documents remain in the project but are excluded from the default Option-C retrieval path.
- Vector backend: local Qdrant in isolated `storage/option_c_qwen3_qdrant`; the existing v6.4.49 Chroma/storage files are not overwritten.
- Embedding: Ollama `qwen3-embedding:8b`. Queries use a technical retrieval instruction; document/passages are embedded without a query instruction.
- Reranker: `BAAI/bge-reranker-v2-m3` through a sigmoid-normalized CrossEncoder confidence scale.
- Generative answers: `qwen2.5:7b` only. Existing deterministic MISRA/direct-grounded fast paths remain and still bypass generation when safe.
- Generated answers receive the full accepted context in the technical profile instead of the compact context previously optimized for the smaller fast model.
- A conservative claim-grounding guard rejects high-confidence unsupported Rule/Directive identifiers, numeric values, multi-token proper names/entities, and acronyms rather than allowing a fluent unsupported answer.
- Exact unsupported identifiers and insufficient evidence retain the exact fallback: `Information not found in company knowledge base.`
- No automatic second LLM verifier is added; the deterministic claim guard preserves latency unless generation is genuinely required.

## Index/build safety

- Option C uses a separate Qdrant/BM25/manifest/cache namespace.
- The v6.4.50 trial intentionally uses a safe **full technical rebuild** rather than attempting an incremental Chroma-to-Qdrant migration.
- Qdrant is built in a staging directory, BM25 is snapshotted, and the active Qdrant directory retains a rollback copy until the companion manifest commit succeeds. A handled failure attempts to restore Qdrant/BM25/manifest state instead of leaving a mixed index.
- Local Qdrant client lifecycles are serialized to avoid competing local-database file locks from Streamlit threads.
- `qdrant-client` is pinned to `1.19.0` in this trial package.

## Office trial sequence

Run in this order with DocuBot stopped during the index build:

1. `Setup_DocuBot_Option_C.bat`
2. `Build_DocuBot_Option_C_Technical_Index.bat`
3. `Run_Option_C_Technical_QA.bat`
4. If the QA result is PASS, `Start_DocuBot_Option_C_Trial.bat`

For rollback, stop the trial and run `Start_DocuBot_Certified_Default.bat`. The old Chroma/E5 storage is intentionally retained.

## Validation status

- Dedicated v6.4.50 Option-C architecture/unit gate: **18/18 PASS**.
- Retained v6.4.39-v6.4.49 rollback hardening: **115/115 PASS** under the explicit legacy profile.
- Broader legacy `tests/` sweep (excluding the new v6.4.50 trial test): **620 PASS / 26 FAIL** across 646 tests. The 26 consist of the same 22 historical/superseded failures carried from prior checkpoints plus 4 deliberately superseded old architecture-policy assertions that still require llama3.2/E5 in the current lock/source text. No unexpected new functional failure was introduced by the Option-C changes.
- Static source validation: **195 Python files / 0 syntax errors**; all **31 BAT files are UTF-8 BOM-free**.
- Legacy `data/all_documents`: **21/21 byte-identical** to v6.4.49. Legacy `storage/`: **123/123 byte-identical** to v6.4.49. The new technical MISRA copy is byte-identical to the original MISRA source PDF.
- Production architecture lock checker: **PASS** for the default v6.4.50 technical trial profile.

Actual Windows/Ollama/Qwen3-Embedding/Qdrant/BGE-reranker-v2-m3 runtime certification is **pending the office-PC run** and must not be inferred from these local/static tests. The dedicated Option-C runtime QA produces a result ZIP for review.

---

# v6.4.49 — Grounded MISRA Yes/No Answer Contract

Built strictly from the certified v6.4.48 checkpoint after office UI testing exposed one answer-format gap. The natural question `Does MISRA allow functions to call themselves?` was semantically and source-grounded correctly to Rule 17.2, but the deterministic MISRA path rendered the full Assessment hierarchy instead of leading with the requested binary answer `No.`.

## v6.4.49 focused recovery

- Genuine MISRA Yes/No questions now preserve their binary answer intent through the specialized MISRA path instead of being overwritten by the generic compliance-assessment focus.
- Source-proven permission questions lead with `Yes.` or `No.` and then one concise Rule/Directive-grounded explanation. Example: recursion permission resolves to `No.` from Rule 17.2.
- Requirement/prohibition questions compare the proposition polarity with the actual source statement, preventing inverted false-Yes answers such as `Does MISRA require recursion?`.
- Visible code compliance/violation questions lead with `Yes.` or `No.` only when deterministic evidence establishes the polarity. Rule-scoped wording is retained; the answer never upgrades one satisfied Rule into whole-code `MISRA-compliant`.
- Uncertain cases such as a Rule 13.5 function call whose persistent side effect is not established continue to answer `Needs more context` rather than guessing Yes or No.
- Natural cue coverage was widened only for wording variants already expressing the same source concept, including plural `functions ... calling themselves` and `macro parameters`; no Rule-number whitelist was added.
- The specialized MISRA generation prompt now carries the same Yes/No contract for guarded generative paths, while the normal reviewer-style `Review this code...` Assessment format remains unchanged.
- No model, embedding, reranker, retrieval threshold, index/chunking, company-document, storage, UI, or latency setting changed.

## Local validation

- Dedicated v6.4.49 YES/NO regression tests: **10/10 PASS**.
- Retained v6.4.39-v6.4.49 hardening block: **115/115 PASS**.
- Pre-lock broad `tests/` regression: **622 PASS / 24 FAIL** across 646 tests = the same **22 known legacy/superseded failures** plus exactly **2 expected architecture-lock/Production-QA hash mismatches** from the three intentional production edits; **0 new functional failures**.
- Final post-lock broad `tests/` regression: **624 PASS / exactly the same 22 known legacy/superseded FAIL** across 646 tests; architecture/Production-QA lock **12/12 PASS**; **0 new functional failures**.
- Static release integrity: **188 Python files / 0 syntax errors**; authoritative MISRA inventory remains **156 Rules + 17 Directives = 173**.
- Generic repository-root pytest collection is not used as a release gate in this build host because the container lacks Streamlit for two legacy dev-script collectors and `test_hash.py` still expects the historical `data/all_documents/policy.md`; this reproduces the known non-production collection issue recorded in earlier checkpoints.
- Windows/Ollama acceptance is intentionally focused: verify natural Yes/No MISRA questions in the actual Streamlit UI while retaining the existing v6.4.48 grounding/citation behavior.

---

# v6.4.48 — Unknown Structured-Reference Latency Closure

Built strictly from the Windows/Ollama-certified v6.4.47 checkpoint after the final 16-case manual/UI validation produced **16/16 semantically correct answers** but exposed one avoidable latency defect. `Explain MISRA Rule 99.99 and give its requirement.` correctly returned the company-KB fallback, yet stale Rule 20.7 follow-up state caused an unnecessary qwen2.5:7b call lasting **16.7085 s** before the fallback completed at **16.7515 s**.

## v6.4.48 focused recovery

- An explicit current-turn structured identifier is now self-contained for follow-up classification; `Rule 99.99`, `Directive 4.12`, or another explicit structured ID cannot inherit a stale prior MISRA scenario merely because the utterance also contains words such as `requirement`.
- Exact Rule/Directive lookup is authoritative. If an explicit Rule/Directive has **no exact structured metadata match**, retrieval stops before semantic/vector/reranker search and before any LLM generation instead of substituting a neighboring rule.
- Section-like misses retain their historical semantic fallback so looser non-MISRA section metadata is not broken.
- `HG-003` and `ADV-011` now fail if the fake-ID answer is the correct fallback but an LLM was unnecessarily invoked; latency regressions can no longer false-PASS on answer text alone.
- The qwen2.5:7b reviewer/code-analysis path is intentionally unchanged. The manual Rule 16.6 switch review remained semantically correct and guarded, so quality is not traded away for artificial speed.
- No model, embedding, reranker, retrieval threshold, context window, keep-alive, index/chunking, company-document, storage, or UI setting changed.

## Manual evidence from v6.4.47

- Final manual/UI batch: **16/16 semantically correct**.
- Median request latency: about **0.064 s**; **10/16** turns were under **0.1 s**.
- Direct MISRA/follow-up paths such as Directive 4.12, Rule 17.2, and rationale follow-ups were about **0.036–0.075 s**.
- Company Leave Policy overview/implicit relation follow-ups were about **3.09–3.23 s** and remained correctly grounded to `Employee Leave Policy.docx`.
- Fake `Rule 99.99` was the only clearly avoidable latency defect at **16.7515 s** because the complex model was invoked before fallback.
- Rule 16.6 reviewer-style switch analysis took **19.4264 s** through qwen2.5:7b and is retained as the intentional complex-quality path.

## Local validation

- Dedicated v6.4.48 latency/grounding tests: **5/5 PASS**.
- Retained v6.4.39-v6.4.48 hardening set: **105/105 PASS**.
- Full pre-lock project regression: **612 PASS / 24 FAIL** = the same **22 known legacy/superseded failures** plus exactly **2 expected architecture-lock/hash mismatches** from intentional v6.4.48 edits; **0 new functional failures**.
- Architecture/Production-QA lock after hash refresh: **12/12 PASS**.
- Frozen full project regression: **614 PASS / exactly the same 22 known legacy/superseded FAIL** across **636 tests**; **0 new functional failures**.
- Static integrity: **187 Python files / 0 syntax errors**; MISRA inventory remains **156 Rules + 17 Directives = 173**; all six QA BAT launchers are BOM-free.
- `config/settings.py`, all **21 company documents**, and all **123 storage files** are byte-identical to v6.4.47.
- Windows/Ollama acceptance is intentionally narrowed to the two affected gates plus one manual fake-ID latency check: Grounded Hallucination Guard **12/12**, Adversarial Grounding **16/16**, then `Rule 99.99` must fall back without an LLM call.

---

# v6.4.47 — Scoped MISRA Compliance-Claim Hardening

Built strictly from v6.4.46 after the actual Windows/Ollama six-gate rerun reached **90/90 official PASS**. Manual semantic audit still found one repeated false-pass wording pattern: answers with source evidence proving only that **Rule 16.6** was satisfied could conclude that the whole switch statement was **MISRA-compliant**. That is broader than the accepted evidence and is therefore not safe enough to lock.

## v6.4.47 focused recovery

- The MISRA semantic alignment guard now rejects blanket `MISRA-compliant` claims when accepted evidence establishes only specific Rule/Directive status.
- Satisfied results remain explicitly scoped to the identified requirement, for example **Rule 16.6: Satisfied / no violation established**, rather than claiming whole-code or whole-construct MISRA compliance.
- LLM Certification, Grounded Hallucination Guard, Question Understanding, and Adversarial Grounding validators now reject this same blanket overclaim so it cannot silently false-PASS again.
- All six Windows/Ollama QA harnesses are version-labeled v6.4.47.
- No LLM/model, embedding, reranker, retrieval threshold, index/chunking, company-document, storage, or UI behavior change is introduced.

## Local validation

- Dedicated v6.4.47 scoped-compliance tests: **4/4 PASS**.
- Retained v6.4.39-v6.4.47 semantic/follow-up/hardening set: **100/100 PASS** before final lock refresh.
- Project regression scope: **631 tests collected**. Pre-lock split result was **607 PASS / 24 FAIL** = the same **22 known legacy/superseded failures** plus exactly **2 expected architecture-lock/Production-QA hash mismatches** from intentional v6.4.47 edits.
- Architecture/Production-QA lock checks after hash refresh: **12/12 PASS**; reconciled frozen result is **609 PASS / exactly the same 22 known legacy/superseded FAIL**, with **0 new functional failures**.
- Static integrity: **186 Python files / 0 syntax errors**; authoritative MISRA inventory remains **156 Rules + 17 Directives = 173**.
- Clean-release retained/architecture validation: **185/185 PASS**.
- Release delta from pristine v6.4.46: exactly **10 intentional files** (9 modified + 1 added test), **0 deletions**.
- `config/settings.py`, **21/21 company documents**, and **123/123 storage files** are byte-identical to v6.4.46; all six QA BATs are BOM-free.

## Windows acceptance

Run the same six QA BATs. Target remains **90/90**, now with tightened scope-overclaim validators. In the Rule 16.6 satisfied cases, the answer must remain requirement-scoped and must not state that the whole switch/code is `MISRA-compliant`. Return the generated result ZIPs and console logs for semantic review before final lock/manual UI acceptance.

---

# v6.4.46 — Windows Closure Recovery

Built strictly from v6.4.45 after the actual Windows/Ollama evidence reached **85/90 official PASS** across six automated gates: LLM Certification **10/13**, Natural Conversation **13/13**, Conversation Robustness **17/18**, Grounded Hallucination Guard **11/12**, Question Understanding **18/18**, and Adversarial Grounding **16/16**. Manual review confirmed three remaining production-quality gaps: a named Employee Leave Policy re-entry could inherit stale MISRA state, one Rule 20.7 why+expansion path collapsed to rationale-only text, and text-only Rule 13.5 uncertainty still referred to a nonexistent "visible snippet". The remaining forced-llama MISRA failure was a certification-role mismatch: production no longer assigns simple MISRA code review to llama3.2:3b.

## v6.4.46 focused recovery

- Explicit named approval subjects such as `Who approves Employee Leave Policy requests?` are no longer classified as deictic grounded follow-ups, so a stale MISRA anchor cannot override the newly named company topic.
- Function-like macro questions containing an actual definition/invocation now bypass rationale-only follow-up finalization and preserve the full Rule 20.7 deterministic assessment, exact literal expansion, and status.
- Text-only Rule 13.5 questions now state that the **question alone** does not establish persistent side effects; they no longer claim a visible snippet/code when none was provided.
- Forced llama3.2 certification is aligned to its real fast company-knowledge role. Source-proven MISRA remains deterministic and reviewer-style MISRA remains qwen2.5:7b. This changes certification validity, not production model routing.
- No LLM/model, embedding, reranker, retrieval-threshold, chunking/indexing, company-document, or storage changes.

## Local validation

- Dedicated v6.4.46 closure suite: **5/5 PASS**.
- Retained v6.4.39-v6.4.46 semantic/follow-up/hardening compatibility before certification-label refresh: **96/96 PASS**.
- Full pre-lock suite: **602 PASS / 24 FAIL** = the same **22 known legacy/superseded failures** plus exactly **2 expected architecture-lock hash mismatches** from the two intentional production edits.
- Final frozen full suite: **605 PASS / exactly the same 22 known legacy/superseded FAIL**; architecture-lock hash failures are gone, with **0 new functional failures**.
- Static integrity: **185 Python files / 0 syntax errors**; MISRA structured inventory remains **156 Rules + 17 Directives = 173**.
- `config/settings.py`, **21/21 company documents**, and **123/123 storage files** are byte-identical to v6.4.45.

## Windows acceptance

Run the same six QA BATs. The target remains **90/90**, but the forced llama case now validates grounded company-knowledge synthesis that matches llama3.2:3b's production responsibility. Return all generated result ZIPs and console logs for actual-answer review before final lock/manual UI acceptance.

---

# v6.4.45 — Deadline Quality + Latency Recovery

Built strictly from v6.4.44 after the actual Windows/Ollama evidence reached **63/74 official PASS** across five automated gates: LLM Certification **10/13**, Natural Conversation **13/13**, Conversation Robustness **14/18**, Grounded Hallucination Guard **11/12**, and Question Understanding **15/18**. Manual semantic review of those logs also found false-pass patterns involving invented user code, contradictory MISRA status wording, policy source-provenance drift, and a hallucinated dynamic-memory Rule identifier. v6.4.45 is a focused recovery for those evidence-backed gaps while protecting both answer quality and latency.

Production settings intentionally unchanged:

- Fast LLM: `llama3.2:3b`
- Complex LLM: `qwen2.5:7b`
- Embedding: `intfloat/multilingual-e5-base`
- Reranker: `BAAI/bge-reranker-base`
- Retrieval thresholds, chunking/index schema, knowledge folder, company-only answer contract, and authoritative MISRA corpus are unchanged.
- MISRA structured inventory remains **156 Rules + 17 Directives = 173** first-class citable references.

Focused quality + latency changes:

- **Dynamic-memory semantic authority:** natural English/Tagalog/Taglish wording such as `dynamic heap memory allocation` is mapped to the authoritative structured MISRA evidence before generic generation, preventing a nearby/cross-reference Rule from becoming the answer identifier. Direct structured facts can finish deterministically.
- **Rule 13.5 natural uncertainty:** short questions about a function call on the right side/right-hand side of `&&` or `||` map to Rule 13.5 and preserve **Needs more context** unless persistent side effects are actually established.
- **Concept vs code separation:** text-only MISRA questions no longer need a fabricated `visible construct` or `user code` narrative. The semantic guard rejects generated source/example code presented as if the user supplied it.
- **Strict status polarity:** a generated answer cannot carry `Potential non-compliance` and later introduce a second confirmed `Violation/Non-compliance` status for the same Rule. Forced-model certification now detects this contradiction instead of false-passing it.
- **Exact macro expansion:** when the user provides a function-like macro invocation and asks for the problematic expansion, the literal substitution is derived deterministically from the shown definition/arguments. No unsolicited parenthesized reconstruction is invented.
- **Policy relation paraphrases:** natural Tagalog/Taglish eligibility and approval forms such as `Sino puwedeng gumamit...` and `Okay, sino naman ang nag-aapprove?` are recognized as relation intent rather than generic prose.
- **Grounded source-drift recovery:** a short same-chat relation follow-up may reuse the previously accepted company source when fresh retrieval completely drifts to an unrelated document. This adds no extra retrieval/model stage.
- **Source-provenance binding:** deterministic policy answers expose sources that contain the actual approval/eligibility facts, preventing a correct Leave Policy answer from being accompanied by an unrelated MISRA source.
- **External-knowledge boundary before retrieval:** explicit requests to use general/external knowledge or facts outside company documents return the exact company-KB fallback **before retrieval and before any LLM call**, improving safety and latency together.
- **Latency-aware deterministic routing:** complete source-provable policy summaries, direct mixed-status MISRA checks, exact macro-expansion checks, and other safely deterministic cases no longer force qwen2.5:7b merely to satisfy a routing expectation. Genuine reviewer-style/narrative reasoning still uses the complex model.
- **Validator quality:** overly phrase-specific checks were relaxed only where a semantically equivalent source-grounded answer is correct; validators were strengthened where contradictions, invented code, wrong source provenance, or fake identifiers must fail.

Windows/Ollama quality gates for v6.4.45:

1. `Run_LLM_Certification_QA.bat` — **13 cases**.
2. `Run_Natural_Conversation_Certification_QA.bat` — **13 turns**.
3. `Run_Conversation_Robustness_QA.bat` — **18 turns**.
4. `Run_Grounded_Hallucination_Guard_QA.bat` — **12 cases**.
5. `Run_Question_Understanding_QA.bat` — **18 cases**.
6. `Run_Adversarial_Grounding_QA.bat` — **16 new adversarial cases** covering misleading paraphrases, fake Rule IDs, external-knowledge pressure, source-provenance drift, invented-user-code behavior, exact macro expansion, status polarity, and company-policy relation grounding.

Total target: **90/90 automated Windows/Ollama checks/turns**. The QA questions are evidence/regression coverage only; production behavior is not implemented as a question whitelist.

Local validation before packaging:

- Dedicated v6.4.45 quality/latency suite: **20/20 PASS**.
- Retained v6.4.39-v6.4.45 semantic/follow-up/hardening compatibility: **91/91 PASS**.
- Production Architecture Lock: **PASS**.
- Full `tests` suite: **600 PASS / exactly the same 22 known legacy/superseded FAIL**; v6.4.44 was **580 PASS / the same 22 FAIL**, therefore **20 additive passes and 0 new functional failures**.
- **184 Python files / 0 syntax errors**.
- All six Windows/Ollama QA BAT launchers are UTF-8 BOM-free.
- `config/settings.py` is byte-identical to v6.4.44.
- `data/all_documents`: **21/21 byte-identical** to v6.4.44.
- `storage`: **123/123 byte-identical** to v6.4.44.
- Authoritative MISRA inventory remains **173** structured references.
- Clean release differs from v6.4.44 by exactly **15 intentional files** and contains **1,249 files** before ZIP packaging.

Windows acceptance remains pending. Run the six QA BAT files in the order listed above. Do not treat v6.4.45 as deployment-certified based only on counters: return the six result ZIPs and six console logs for semantic review of the actual answers. Only after all six gates and actual-answer review are clean should the larger normal-UI manual natural-conversation batch be run.

---

# v6.4.44 — Deep Conversation, Question Understanding + Hallucination Hardening

Built strictly from v6.4.43 after actual Windows/Ollama evidence showed that the original 13-case LLM gate reported **13/13 PASS** while manual semantic review still found false-pass status contradictions, and the focused natural-conversation gate reported **10/13 PASS** with a hidden stale-macro-anchor false pass. This release closes those evidence-backed gaps and expands certification depth without changing the production model, embedding, reranker, retrieval threshold, chunking, knowledge-folder, company-only answer contract, or the authoritative **156 Rules + 17 Directives = 173** MISRA structured references.

Focused production changes:

- **Latest code/snippet authority:** a fresh inline preprocessor definition such as `Paano naman kung ganito? #define SQUARE(x) ((x) * (x))` is recognized as new code instead of inheriting the previous unsafe macro scenario.
- **Rule 20.7 protected-macro assessment:** visibly parenthesized function-like macro parameters can be assessed as satisfied even when the user supplies only the corrected definition and no invocation. An immediate `May kailangan pa ba akong baguhin diyan?` follow-up stays on that latest protected macro rather than reviving the previous unsafe expansion.
- **Source-exact rationale follow-ups:** short `Bakit?/Why?` MISRA follow-ups can return the accepted source rationale directly; Rule 17.2 therefore preserves the actual `stack space` / `worst-case stack usage` rationale instead of replacing it with a looser generic hazard.
- **Compact policy overview completeness:** broad requests such as `Tell me about the Employee Leave Policy` preserve all compact labeled source facts when available, including vacation leave, sick leave, approval, and eligibility, instead of dropping the final relation.
- **Stricter status-polarity verification:** generated MISRA answers must keep one canonical status per Rule/Directive. A `Needs more context` item cannot later become `Potential non-compliance`, and a deterministic `Potential non-compliance` cannot be escalated to confirmed `Non-compliance` or contradicted by a second status heading.
- **Stricter forced-model certification:** the forced llama3.2:3b and qwen2.5:7b certification cases now require the actual generated answer to pass semantic alignment; a contradictory model answer can no longer receive a certification PASS merely because a later safety path could repair it.
- **Focused natural gate tightened:** the existing 4-flow / 13-turn gate now checks exact Rule 17.2 rationale wording, protected Rule 20.7 status, and the immediate post-correction follow-up so the v6.4.43 hidden stale-anchor false pass is explicitly caught.

Expanded Windows/Ollama quality gates:

1. `Run_LLM_Certification_QA.bat` — **13 cases**: deterministic/forced/hybrid routing, grounded LLM generation, same-document policy relations, MISRA status polarity, literal macro handling, and fallback safety.
2. `Run_Natural_Conversation_Certification_QA.bat` — **13 turns**: focused same-chat dynamic-memory, topic-switch, recursion, and macro-history regressions.
3. `Run_Conversation_Robustness_QA.bat` — **18 turns**: return-to-old-topic behavior, multi-turn policy relations, cross-domain switching, structured follow-up rationale/scope, and fresh-chat history isolation.
4. `Run_Grounded_Hallucination_Guard_QA.bat` — **12 independent cases**: unsupported/current external knowledge, nonexistent MISRA IDs, literal/protected macro grounding, mixed status polarity, exact code-line grounding, compliant-construct restraint, dynamic-memory multi-reference coverage, and policy completeness.
5. `Run_Question_Understanding_QA.bat` — **18 independent cases**: English/Tagalog/Taglish paraphrases, natural MISRA concept mapping, eligibility vs approval vs quantity intent, exact code grounding, protected macro understanding, and unresolved/external fallback behavior.

That is **74 automated Windows/Ollama checks/turns** across five complementary gates. These QA prompts are regression/generalization evidence only; production behavior is not implemented as a question whitelist.

Local validation before packaging: dedicated v6.4.44 hardening suite **16/16 PASS**; retained v6.4.39-v6.4.44 semantic/follow-up/hardening set **71/71 PASS**; Production Architecture Lock/Production-QA compatibility checks **12/12 PASS**. Full `tests` regression is **580 PASS / exactly the same 22 known legacy FAIL** versus v6.4.43 **564 PASS / the same 22 FAIL**, giving **16 additive passes and 0 new functional failures**. **182 Python files** parse with **0 syntax errors**. `config/settings.py`, all **21 company documents**, and all **123 storage files** are byte-identical to v6.4.43 before the clean release copy, and the authoritative MISRA inventory remains **156 Rules + 17 Directives = 173**. Fresh Windows/Ollama execution of all five gates is still required before v6.4.44 can be called deployment-certified.

Recommended Windows validation order before manual UI testing:

1. `Run_LLM_Certification_QA.bat` — target **13/13 PASS**.
2. `Run_Natural_Conversation_Certification_QA.bat` — target **13/13 PASS**.
3. `Run_Conversation_Robustness_QA.bat` — target **18/18 PASS**.
4. `Run_Grounded_Hallucination_Guard_QA.bat` — target **12/12 PASS**.
5. `Run_Question_Understanding_QA.bat` — target **18/18 PASS**.

Only after all five automated gates are clean should the larger normal-UI manual natural-conversation batch be used as the final human presentation/behavior check.

---

# v6.4.43 — Natural Conversation + Certification Recovery

Built strictly from v6.4.42 after the actual Windows/Ollama certification reached **10/13 PASS** and the same-session natural-conversation evidence exposed broader topic/referent regressions. The recovery is intentionally focused: no model, embedding, reranker, retrieval threshold, chunking, knowledge-folder, fallback-contract, or company-knowledge-only policy change. The authoritative MISRA inventory remains **156 Rules + 17 Directives = 173 structured references**.

Focused changes:

- **Explicit topic override recovery:** a clearly named multi-word subject such as `Employee Leave Policy` now overrides a stale prior topic such as MISRA, while pronoun/deictic follow-ups still resolve against the latest valid referent.
- **Grounded follow-up anchor recovery:** `dito/iyan/it/that/siya/niya` and short relation follow-ups can reuse the latest accepted grounded request, but fresh pasted code or a fresh MISRA semantic scenario does not inherit an older code scenario.
- **Natural MISRA semantic mapping:** recursion/self-call and dynamic-memory wording is matched against the authoritative structured Rule/Directive bodies before general-corpus interpretation; plural/applicable dynamic-memory requests can surface every independently supported structured requirement.
- **Structured-reference authority:** Rule/Directive identity remains citable only from authoritative structured MISRA records, preventing appendix/rationale/cross-reference mentions from becoming the primary answer identifier.
- **Eligibility vs approval hardening:** same-document composition now scores the explicit Eligibility/Qualification field above nearby leave-day entitlement quantities, and natural `Who can use it?` wording resolves only when accepted company context exposes an explicit eligibility relation.
- **Grounded scope follow-up:** narrow questions such as `Does it apply indirectly too?` can answer directly from already accepted source text when the requested scope word is literally present.
- **Mixed-status LLM protection:** generation normalization can restore a missing canonical status label only when it does not contradict the model's wording; semantic alignment still rejects reversed polarity. This protects mixed Rule 13.3 / Rule 13.5 answers without hiding a genuinely wrong assessment.
- **Rule 20.7 literal-expansion protection:** unsolicited model-created parenthesized reconstructions are stripped/rejected when the user asked for analysis rather than remediation; the literal visible substitution remains authoritative.
- Added `Run_Natural_Conversation_Certification_QA.bat`, a second one-click Windows/Ollama gate with **4 isolated flows / 13 same-chat turns** covering the exact natural-conversation regressions found in v6.4.42. The original LLM-inclusive certification remains **13 cases**.
- Added `qa/v6_4_43_natural_conversation_certification_recovery.txt` with the Windows run order and optional manual UI spot-check flows.

Local validation before packaging: dedicated v6.4.43 recovery tests **16/16 PASS**; retained v6.4.40-v6.4.43 plus architecture/Production-QA compatibility set **60/60 PASS**; full `tests` suite **564 PASS / the same 22 known legacy FAIL** versus v6.4.42 **548 PASS / the same 22 FAIL**, so **0 new functional failures**. Production Architecture Lock validation is **5/5 PASS**. **178 Python files** compile with **0 syntax errors**. `config/settings.py`, all **21 company documents**, and all **123 storage files** remain byte-identical to v6.4.42 in the clean release build. The generic repository-wide `pytest` collection still has the same three legacy/environment collection issues: two dev scripts require Streamlit in this Linux build container, and legacy `test_hash.py` expects `data/all_documents/policy.md`. The clean release differs from v6.4.42 by only **11 intentional files**, contains **1,237 files**, and preserves `config/settings.py`, all **21 company documents**, and all **123 storage files** byte-for-byte. Fresh Windows/Ollama acceptance remains required: first run `Run_LLM_Certification_QA.bat` (target **13/13 PASS**), then `Run_Natural_Conversation_Certification_QA.bat` (target **13/13 PASS**).

---

# v6.4.42 — Grounded LLM Contract + History Isolation + Same-Document Scope Recovery

Built strictly from v6.4.41 after the actual Windows/Ollama LLM-inclusive certification reached **8/11 PASS**. The three remaining certification failures were all in nuanced grounded MISRA generation: mixed Rule 13.3/13.5 status preservation, Rule 20.7 macro remediation hallucination, and text-only Rule 13.5 uncertainty preservation. Manual company-policy testing also showed that approval and eligibility could each be retrieved correctly while a same-document scope follow-up still fell back. Production model, embedding, reranker, retrieval threshold, chunking, company-knowledge-only policy, and the complete MISRA inventory remain unchanged.

Focused changes:

- Added an **evidence-derived MISRA generation contract**. Every structured Rule/Directive sent to the LLM now carries its already-grounded required status (`POTENTIAL NON-COMPLIANCE`, `NEEDS MORE CONTEXT`, or `SATISFIED / NO VIOLATION ESTABLISHED`) plus the deterministic visible-code observation. The LLM may explain naturally but may not reverse those states.
- Fresh self-contained MISRA questions now use **generation-history isolation**: prior user code is not exposed to the LLM for a new code/scenario review. True grounded deictic follow-ups still preserve the immediately accepted context. This prevents unrelated earlier constructs such as `malloc/free` from contaminating a later switch/code review.
- Rule 20.7 macro handling now carries the **literal visible expansion** into the generation contract and forbids unsolicited invented corrected code. A semantic safety guard also rejects a suggested macro fix that leaves affected parameter expansions unparenthesized.
- If a nuanced MISRA LLM draft fails the semantic guard, DocuBot performs **one grounded LLM self-repair attempt** using the same accepted evidence and guard feedback. Only a repaired answer that passes both reference grounding and semantic alignment is shown; otherwise the certified deterministic grounded finalizer remains the safety fallback.
- Added a domain-neutral **same-document approval + eligibility scope finalizer**. It only composes an answer when both relations are explicitly present in the accepted company context and does not infer that an approver is the eligible population.
- `Run_LLM_Certification_QA.bat` is expanded from **11 to 13 cases**. New coverage proves same-document approval/eligibility scope behavior and fresh MISRA generation-history isolation while preserving actual `qwen2.5:7b` use.
- **All MISRA Rules and Directives remain first-class.** The authoritative structured inventory is still **156 Rules + 17 Directives = 173 citable references**. The new contract is generated dynamically from retrieved evidence; it is not a Rule/Directive whitelist.

Local validation before packaging: dedicated v6.4.42 tests **12/12 PASS**; retained focused v6.4.29-v6.4.42 MISRA/LLM regression set **94/94 PASS**. Full `tests` suite after updating the production architecture lock for the intentional changed critical files: **548 PASS / the same 22 known legacy FAIL**, with **0 new functional failures**. The generic repository-wide `pytest` collection still has the same environment/legacy collection issues documented in prior releases (dev scripts require Streamlit in the build container; legacy `test_hash.py` expects `data/all_documents/policy.md`). **176 Python files** compile with **0 syntax errors**. `config/settings.py`, the production model/retrieval policy, all **21 company documents**, and all **123 storage files** are unchanged from v6.4.41 after restoring the test-mutated manifest.

Fresh Windows/Ollama certification is still required before v6.4.42 can be called deployment-certified. Run `Run_LLM_Certification_QA.bat`, then use `qa/v6_4_42_grounded_llm_contract_manual_validation.txt` for the manual unseen checks.

---

# v6.4.41 — MISRA Semantic/Anchor/Relation Hardening

Built strictly from v6.4.40 after the actual Windows/Ollama evidence reached 7/8 PASS and manual review exposed four concrete quality gaps: incorrect Rule 20.7 macro-expansion reasoning, follow-up anchor drift into an older Rule 17.4 scenario, eligibility being collapsed to the nearby approver role, and Rule 18.4 highlighting the declaration instead of the actual pointer-arithmetic line. Production model/retrieval/index settings remain unchanged.

Focused changes:

- Rule 20.7 now keeps literal function-like macro substitution grounded in the visible definition/arguments. For `#define SQUARE(x) x * x` with `SQUARE(a + b)`, the visible expansion path is preserved as `a + b * a + b`; inserted parentheses are rejected by the semantic guard.
- The MISRA semantic alignment guard now validates status polarity more strictly, so wording such as `does not violate` cannot accidentally satisfy an expected non-compliance just because it contains the word `violate`. It also rejects mismatched explicit macro-expansion claims.
- Text-only but self-contained MISRA scenarios can establish a new semantic anchor instead of inheriting an unrelated older code/scenario merely because the wording contains `iyon/that/it`. True deictic follow-ups such as `Bakit mo nasabi yan?` still reuse the prior grounded scenario.
- Current-turn eligibility/entitlement intent outranks approval words carried only by prior grounded context. A deterministic compact eligibility finalizer extracts the explicit eligibility/entitlement source value instead of returning a nearby approver.
- Rule 18.4 visible-code assessment now points to the actual pointer-arithmetic statement such as `p = p + 2;`, while Rule 18.1 can still be independently satisfied for the same in-range pointer result.
- The LLM certification harness is expanded to 11 cases, adding direct coverage for eligibility-vs-approver separation, literal Rule 20.7 macro expansion, and text-only Rule 13.5 right-hand-side function-call reasoning.
- **All MISRA Rules and Directives remain in scope.** The authoritative structured inventory is still validated as **156 Rules + 17 Directives = 173 citable Rule/Directive bodies**. The new logic adds semantic interpretation/guardrails only; it does not whitelist the test cases or reduce corpus coverage.

Local validation before packaging: dedicated v6.4.41 tests **9/9 PASS**; selected retained v6.4.29-v6.4.40 MISRA compatibility plus v6.4.41 checks **75/75 PASS**. Full `tests` suite: **536 PASS / the same 22 known legacy FAIL**, with **0 new functional failures**. **175 Python files** compile with 0 syntax errors. `data/all_documents` remains **21/21 byte-identical** and `storage` **123/123 byte-identical** to v6.4.40 after restoring the test-mutated manifest. `config/settings.py` is byte-identical to v6.4.40.

Fresh Windows/Ollama certification is still required before this checkpoint can be called deployment-certified. Run `Run_LLM_Certification_QA.bat`, then repeat the manual follow-up sequence in `qa/v6_4_41_semantic_anchor_relation_hardening.txt`.

---

# v6.4.40 — Grounded LLM Semantic Alignment + Follow-Up Recovery

Built strictly from v6.4.39 after actual Windows/Ollama LLM certification proved both local models were active but exposed one routing failure plus semantic/follow-up quality gaps. Production model, embedding, reranker, retrieval threshold, chunking, and company-knowledge-only policy remain unchanged.

Focused changes:

- Multi-point company-policy summary requests now outrank narrow approval/approver extraction, so a request covering vacation leave, sick leave, approval, and eligibility routes to grounded complex synthesis instead of collapsing to `manager`.
- Natural MISRA reviewer/explanation questions can use grounded `qwen2.5:7b` synthesis when reasoning adds value, while terse source-proven checks keep the deterministic low-latency safety path.
- A semantic MISRA alignment guard now validates not only Rule/Directive citation grounding but also whether generated violation/satisfied/needs-more-context states agree with the visible code and source-proven interpretation. Unsafe LLM assessments are rejected instead of being shown.
- Same-chat grounded follow-up state preserves the previous accepted question/code, evidence, sources, and assessment context for conservative follow-ups such as `Bakit?`, `aling part mismo ng code?`, `same document?`, and supported-rule follow-ups.
- Boolean-return function conditions such as `bool_t ready(void); while (!ready())` are interpreted against the structured MISRA corpus instead of falling through to unrelated generic retrieval.
- Rule 11.9 observation wording is bound to the integer null-pointer initialization (`ptr = 0`) rather than incorrectly attributing that rule to `if (ptr)`.
- `Run_LLM_Certification_QA.bat` is upgraded for v6.4.40 and now includes normal-production complex MISRA routes plus semantic-alignment verification.

Local validation before packaging: dedicated v6.4.40 tests **11/11 PASS**. Full suite: **527 PASS / the same 22 known legacy FAIL**, versus v6.4.39 baseline **516 PASS / the same 22 FAIL**, so there are **0 new functional failures** and 11 additive passes. **174 Python files** compile with 0 syntax errors. `data/all_documents` remains **21/21 byte-identical** and `storage` **123/123 byte-identical** to v6.4.39 before packaging. Actual Windows/Ollama LLM certification remains required before this checkpoint can be called deployment-certified.

---

# v6.4.39 — LLM-Inclusive Certification + Clean MISRA Responses

Built strictly from v6.4.38. Production model/retrieval policy is unchanged.

This checkpoint adds two focused upgrades:

- Clean natural MISRA presentation: no internal HTML metadata comments, no bare-number rationale such as `Rationale: 1.`, and more code-specific `Why it applies` wording. Exact Rule/Directive explanation requests keep the certified source-shaped format.
- `Run_LLM_Certification_QA.bat`: a one-click Windows/Ollama certification harness that proves actual grounded use of `llama3.2:3b`, `qwen2.5:7b`, normal hybrid fast/complex routing, deterministic safety routing, and safe fallback for unsupported company facts.

The certification-only environment gate `DOCUBOT_LLM_CERT_FORCE_GENERATION=1` is OFF by default and is used only by the QA harness to bypass deterministic MISRA finalization after grounded retrieval. Normal production behavior remains deterministic-first where safe, with LLM reasoning used when routing requires it.

LLM certification evidence is written under `logs\llm_certification`.

---

# v6.4.24 — Source-Exact Responsibility Finalization Recovery

Built strictly from v6.4.23 after the actual Windows Production QA rerun reached 9/10 PASS. The only remaining failure was the runtime unseen-document P-003 case: the correct temporary source was retrieved, but the generated answer changed the exact source role suffix from `02O0` to `02O`.

This release makes a narrow, domain-neutral recovery:
- `Who owns ...?` is now explicitly classified as a responsibility/ownership intent.
- When accepted company context contains an explicit ownership/responsibility key/value label lexically tied to the question (for example `Escalation owner role: <value>`), DocuBot returns the source value deterministically before LLM generation.
- The extracted value preserves opaque alphanumeric IDs/codes exactly; no document, role name, QA token, or expected answer is hardcoded.
- Models, embedding/reranker, retrieval threshold, hybrid weights, chunking, index schema, KB folder, citations, fallback contract, and the selected hybrid 3B/7B policy remain unchanged.

Windows Production QA must be rerun before this recovery is certified.

---

# v6.4.23 — Production QA Recovery

This checkpoint is a focused recovery from the actual Windows v6.4.22 Production QA evidence. Eight of ten Production QA stages passed; the two failures were isolated to one QA-harness classification bug and one genuine mixed-language relation-retrieval gap.

Changes are intentionally narrow:

- `scripts/run_production_document_lifecycle.py` now checks smart-build add/update/delete classification using manifest metadata and case-insensitive filename identity. This fixes the Windows `os.path.normcase()` false negative where random mixed-case QA filenames were lowercased in manifest document keys even though add/update/delete behavior and KB restoration were correct.
- `services/answer_service.py` now treats strong Tagalog source/relation phrases such as `ayon sa`, `tungkol sa`, `para sa`, and `mula sa` as multilingual cues even when the question starts with an English interrogative.
- `chat/query_enricher.py` generically extracts the real subject from front-loaded approval wording such as `approval role/authority ... ayon sa / according to / under / in / for ...`, preventing relation enrichment from anchoring to a determiner such as `the`.

No model, embedding model, reranker model, retrieval threshold, vector/BM25 weights, chunking, index schema, knowledge folder, fallback contract, or hybrid 3B/7B routing policy is changed. The Production Architecture Lock manifest is advanced to v6.4.23 only to record the two intentional critical-file hashes.

Windows acceptance: run `Run_Production_QA.bat` with the existing `data\all_documents`, `storage`, and virtual environment. Target: `PRODUCTION QA: PASS (10/10 stages)`. See `qa/v6_4_23_production_qa_recovery.txt`.

# v6.4.21 — Production Architecture Lock

This checkpoint freezes the certified DocuBot production architecture after the v6.4.20 model-policy A/B decision. It introduces **no production RAG/runtime behavior change**. The selected production policy remains `llama3.2:3b` for fast/default generation and `qwen2.5:7b` for complex generation; the unified 3B candidate is explicitly rejected because it scored 77/78 despite lower latency.

`PRODUCTION_ARCHITECTURE_LOCK.json` records the selected model/retrieval/indexing/answer contract and SHA-256 hashes for critical production files. `Run_Production_Architecture_Lock_Check.bat` validates those settings, rejects an active QA-only forced-model override, and detects critical-file drift before Production QA or deployment packaging. See `qa/v6_4_21_production_architecture_lock.txt`.

# v6.4.20 — Model-Policy A/B Validation

- Validation-only checkpoint; production runtime behavior remains v6.4.18-certified behavior.
- Adds one-click `Run_Model_Policy_AB_Validation.bat`.
- A/control: `llama3.2:3b` fast + `qwen2.5:7b` complex.
- B/candidate: router unchanged, but both fast and complex routes use `llama3.2:3b`.
- Candidate is eligible only with exact 78/78 QA plus 4/4 unseen-document proof and cleanup PASS.
- No automatic production policy change, rebuild, rechunk, or dependency install.

# v6.4.19 - Post-Optimization Certification Harness

Built from v6.4.18 with **no production RAG/runtime behavior change**. This checkpoint adds realistic 0s/3s/6s/9s Qwen-to-Llama follow-up timing and a one-click post-optimization certification runner that combines the General Corpus Gate with latency evidence. See `qa/v6_4_19_post_optimization_certification.txt`.

## v6.4.18 Post-Complex Fast-Model Residency Recovery

v6.4.18 is the second quality-neutral latency checkpoint built directly on v6.4.17. Actual Windows/Ollama evidence confirmed startup prewarm reduced the identical first semantic multilingual query from 29.401s to 9.450s (67.9%), while the remaining major transition penalty was Qwen2.5:7B -> Llama3.2:3B: the first fast query after a complex-model turn took 14.328s, then stabilized at 4.599s and 4.421s once the fast model was resident again. This release therefore preserves the certified routing, prompts, retrieval ranking, 0.55 gate, chunking, model identities, embeddings and reranker, and changes only Ollama residency orchestration. After all complex-answer generation and verification is complete, a bounded daemon recovery requests a load-only restore of the existing fast model while the user reads the answer. If the next fast query arrives while that restore is still in flight, it reuses/waits for the same load instead of racing a duplicate Ollama switch. Auto mode requires at least 12 GB total RAM and 3 GB currently available; it can be disabled with `DOCUBOT_POST_COMPLEX_FAST_RECOVERY=off`. Startup prewarm from v6.4.17 remains unchanged. No pip install, rechunk, or KB rebuild is required.

Local validation: 14/14 focused v6.4.17/v6.4.18 runtime tests pass; 383/383 release-relevant current tests pass with warnings-as-errors when the already rejected v6.4.14.2.5 branch test and superseded v6.4.10 sidebar-boundary test are excluded.

## v6.4.17 Low-Spec Startup Prewarm Optimization

v6.4.17 is the first performance optimization checkpoint built on the Windows/Ollama-certified v6.4.15.9 general-corpus behavior and the measured v6.4.16 low-spec baseline. The baseline showed the first semantic multilingual request spent 13.282s loading the embedding/vector stack and 2.612s loading the CrossEncoder before normal retrieval/generation, while the exact structured path remained ~0.055s. This release therefore keeps retrieval ranking, source acceptance, chunking, models, embeddings, reranker model/threshold, prompts, and answer logic unchanged, and moves only runtime initialization work earlier. When the KB is healthy, Streamlit starts a bounded background warmup for the exact cached Chroma/embedding/reranker resources plus a load-only Ollama request for the existing llama3.2:3b fast model. Auto mode activates only when detected physical RAM is at least 12 GB and at least 4 GB is currently available; `DOCUBOT_STARTUP_PREWARM=off` disables it and `=on` forces it. A semantic query that arrives while retrieval warmup is already running waits for/reuses that one cached initialization instead of racing a duplicate load. The latency benchmark now includes the same multilingual question both truly cold and after explicit startup prewarm, so Windows evidence can measure the real first-question reduction. No pip install, rechunk, or KB rebuild is required.

Local validation: 18/18 focused startup-prewarm/baseline/parser tests pass; 376/376 release-relevant current tests pass with warnings-as-errors when the already rejected v6.4.14.2.5 branch test and superseded v6.4.10 sidebar-boundary test are excluded; all 132 Python files compile.

## v6.4.16 Low-Spec Windows Baseline Instrumentation

v6.4.16 is a measurement-only checkpoint built on the Windows/Ollama-certified v6.4.15.9 general-corpus release. It does not change production retrieval, answer generation, chunking, models, embeddings, reranker scoring, confidence thresholds, index schema, UI, or knowledge-base contents. It adds `Run_Low_Spec_Latency_Benchmark.bat` plus dependency-free benchmark helpers that capture PC CPU/RAM details, current Ollama model residency, process-cold structured/fast/complex requests, same-process warm requests, repeated warm-query stability, and component-level retriever/vector/reranker/LLM timings. Results are written under `logs/low_spec_latency/run_*` as TXT/JSON/CSV so latency optimization can be based on actual lower-spec Windows evidence rather than inferred from the development machine. Existing loaded Ollama models are recorded but are not forcibly stopped; for the cleanest cold baseline, run the benchmark before opening DocuBot after a reboot. No `pip install`, rechunk, or KB rebuild is required.

## v6.4.15.9 Structured Reference False-Positive Recovery

v6.4.15.9 is built directly from the user's latest project ZIP after Windows evidence showed the final unseen-document proof failure was not caused by chunking, translation, or reranking. The exact temporary TXT was already BM25 rank #1 with the required numeric fact, but `spare parts` was falsely parsed as the structured identifier `Part s`. The structured-ID consistency guard then rejected all candidates before reranking/rescue. This release hardens the generic structured-reference parser by requiring a true word boundary after Section/Article/Chapter/Part so ordinary plural/prose words such as `parts` cannot become fake structured references. Genuine references such as `Part 2`, `Part IV`, `Article 5`, `Chapter 3`, and `Section 6` remain supported. No chunking, model, embedding, reranker, threshold, index schema, or certified retrieval-path change is made.

Local validation on the uploaded latest project: 47/47 focused structured-reference tests pass; 364/364 release-relevant current tests pass with warnings-as-errors when the already rejected v6.4.14.2.5 branch test file and superseded v6.4.10 sidebar-boundary test file are excluded. Both excluded files were confirmed to fail identically on the pristine uploaded project before this change, so they are not regressions introduced by v6.4.15.9. All 125 Python files compile successfully.

## v6.4.15.8 Multilingual General Corpus Rescue

v6.4.15.8 is a focused follow-up to the Windows-validated v6.4.15.7 gate evidence. Gate 1 was HEALTHY and Gate 2 independently verified 78/78 PASS, while the runtime unseen-document proof reached 3/4: three English questions over a newly added random TXT passed with correct source attribution, but one Tagalog numeric/facet question fell back despite the same source containing the answer. The general-corpus rescue already discovered candidates using the canonical multilingual intent query, but reranked those candidates using the original mixed-language surface query. This release makes rescue discovery and rescue reranking use the same canonical semantic query. The change remains fallback-only, keeps the certified primary path untouched, and does not change models, embeddings, reranker, MIN_RETRIEVAL_SCORE, index schema, or chunking.

## v6.4.15.7 Artifact-Verified General Corpus Gate

v6.4.15.7 is a validation-harness-only follow-up to v6.4.15.6. Actual Windows/Ollama evidence on 2026-09-13 showed the product baseline itself reached 78/78 PASS with KB Health HEALTHY, but `Run_General_Corpus_Gate.bat` still stopped at Gate 2 because the long-running QA Python producer returned a non-zero process code after already writing a complete valid PASS artifact. This release leaves production RAG behavior untouched and independently verifies fresh structured artifacts created after per-stage marker files. Gate 2 requires exactly 78/78 PASS with 0 FAIL/ERROR from a fresh `results.json`; Gate 3 requires 4/4 PASS, all source checks true, cleanup success, and `overall=PASS` from a fresh unseen-document proof JSON. Producer exit codes are warnings only when those fresh artifacts independently certify success.

## v6.4.15.6 Deterministic Routing-Order + Infobox/Temporal Record-Scan Recovery

v6.4.15.6 is based strictly on v6.4.15.5 and responds to the actual Windows/Ollama gate evidence from 2026-09-13. KB Health remained HEALTHY and the certified baseline remained 76/78, with only R-006 and R-015 failing. This build does not redesign retrieval. It fixes two deterministic finalization/routing gaps: (1) role/position association can now recover compact infobox/table labels that appear before the named subject, and explicit role-association questions are finalized deterministically before model generation when accepted context proves the relation; (2) temporal BM25 retry no longer exits early when top-K adds no new chunk, and may scan the already-loaded BM25 records for full-date + event-anchor coverage before passing candidates through the existing strict two-event temporal finalizer. Models, embeddings, reranker, confidence threshold, index format, corpus rules, and fallback behavior remain unchanged.

Local validation for this package includes focused regression over the exact Windows failure shapes, the prior temporal recovery suites, warnings-as-errors full regression, compile checks, ZIP integrity, and clean re-extract reruns. Actual Windows/Ollama certification still requires the user's `Run_General_Corpus_Gate.bat` result.

## v6.4.15.5 Subject-Bound Role + Temporal BM25 Grounding Recovery

v6.4.15.5 is based on v6.4.15.4 and responds directly to the actual Windows/Ollama gate evidence from 2026-09-13: KB Health remained HEALTHY and the 78-turn baseline improved to 76/78, leaving only two real product blockers. R-006 returned the nearby phrase `Brains` instead of the requested Bonifacio role, while R-015 returned only `June 1898` because accepted context did not contain enough grounded evidence to finalize both compared dates.

Focused recovery:

- Generic role/position association now requires an explicit subject-to-role relation in accepted source text. Supported source shapes include `SUBJECT as ROLE`, `SUBJECT was named/elected/appointed ROLE`, `installed SUBJECT as ROLE`, subject-led member/profile clauses such as `SUBJECT (...) - the third ROLE (...) of SCOPE`, and an immediately following pronoun sentence tied directly to the subject. A role or nickname belonging to another person can no longer be borrowed merely because it appears nearby.
- When a query explicitly asks for a top/highest/supreme role, directly subject-bound apex-role candidates receive a generic organizational-hierarchy preference (for example president/chief/head/chair/director). No person, document, historical topic, or expected answer is hardcoded.
- The locked v6.4.14.2.4 temporal finalizer remains first authority. If accepted context still cannot bind both events, a new BM25-only temporal grounding retry searches the already indexed corpus separately for the two query facets. Supplemental chunks are used only to attempt the strict deterministic two-date finalizer; they are not sent to the LLM unless deterministic grounding succeeds.
- Successful temporal retry keeps only source chunks containing dates actually used in the deterministic answer, preserving concise source traceability.
- v6.4.15.4 evaluator localization support remains unchanged, and v6.4.15.3 corpus rescue remains fallback-only.

No model, embedding, reranker, global 0.55 threshold, chunk/index schema, requirements, or storage-format change is included. Keep the existing `data/all_documents`, `storage`, and virtual environment. No `pip install` and no forced Knowledge Base rebuild are required.

Local regression before packaging: 348/348 tests pass with warnings treated as errors.

## v6.4.15.4 Certified Focus + Temporal Guard Recovery

v6.4.15.4 is based on v6.4.15.3 and responds directly to the actual Windows gate evidence from 2026-09-13. KB Health remained healthy and the fallback-only general-corpus rescue recovered the earlier R-005 and R-047 failures, but the certified baseline still reported three items: one real role/focus error (R-006), one real temporal regression (R-015), and one evaluator-only localization false negative (R-012).

Focused recovery:

- The exact v6.4.14.2.4 deterministic temporal comparison finalizer is now the first authority for all wording it already certified. The broader v6.4.15 compare/chronological/date-layout logic is fallback-only. New temporal generality therefore cannot override a previously certified Cycle-1 answer.
- Narrow role/position questions of the generic form `which/what <scope> position/role/title is associated/connected with <subject>` can be finalized from explicit accepted source grammar such as `ROLE of SCOPE`, `the third ROLE (...) of SCOPE`, or `was ... ROLE (...) of SCOPE`. This is domain-neutral and uses only already accepted company context; no person, document, historical topic, or expected answer is hardcoded.
- The robustness evaluator accepts `España` as a valid localized spelling for Spain in the Treaty-parties assertion. This is evaluator-only and does not alter production retrieval or answers.
- The v6.4.15.3 certified-path-first retrieval policy remains unchanged: ordinary successful retrieval uses the locked path; corpus-wide lexical rescue activates only when that path would otherwise produce no accepted context, and rescued chunks must still pass the unchanged 0.55 reranker threshold.

No model, embedding, reranker, global threshold, chunk/index schema, requirements, or storage format change is included. Keep the existing `data/all_documents`, `storage`, and virtual environment. No `pip install` and no forced Knowledge Base rebuild are required.

## v6.4.15.3 Certified-Path-First General Corpus Rescue

v6.4.15.3 responds to the actual Windows v6.4.15.2 gate result: KB Health is now healthy, but the certified 78-turn baseline remained 75/78 with the same R-006, R-005, and R-047 regressions. The always-on corpus candidate blending introduced in v6.4.15 is therefore retired from the normal retrieval path.

Production retrieval policy:

- Normal successful queries now follow the v6.4.14.2.4-certified retrieval order, candidate pool, reranker threshold, confidence gate, and diversity path. Corpus-wide lexical discovery does not pre-merge, reshuffle, or reserve reranker slots for those queries.
- General-corpus support remains additive: only when the certified path would otherwise return no accepted context does DocuBot run a second-pass corpus-wide lexical discovery over the already loaded BM25 records.
- Second-pass candidates are independently reranked by the existing CrossEncoder and must still pass the unchanged global 0.55 confidence threshold before they can enter answer context. No weak lexical candidate can bypass the reranker when reranking is enabled.
- Generic list/table intent remains domain-neutral, so explicit employee/procedure/list questions are not scored by relationship/biography keywords.
- Generic temporal comparison forms added in v6.4.15 remain supported without changing the primary certified retrieval order.
- KB Health keeps intentional minimum-length skips visible but non-blocking; missing files, unsupported failures, zero-chunk indexed files, and non-benign skips remain blocking.

No model, embedding, reranker, global threshold, chunk/index schema, or requirements change is included. Existing `storage` should be kept; no `pip install` and no forced full Knowledge Base rebuild are required.

Local regression before packaging: 342/342 tests pass with warnings treated as errors; 116 Python files compile cleanly.

## v6.4.15.2 Evidence-Preserving General Corpus Gate Recovery

v6.4.15.2 is based on v6.4.15.1 and responds directly to the actual Windows validation evidence from 2026-09-13. The production goal remains unchanged: `data/all_documents` is the knowledge base, and arbitrary supported indexed documents must be retrievable without topic-, filename-, entity-, or test-specific hardcoding.

Windows evidence recovery:

- The v6.4.15 corpus-wide lexical channel no longer reshuffles the already ranked BM25/vector candidate list using synthetic coverage scores. Existing candidates keep the certified base order; coverage-only candidates are appended as rescue evidence.
- The pre-reranker source-diversity policy is now evidence-preserving. The strongest base-ranked prefix is protected, and a different source can replace only a weak tail slot when it is a coverage-only candidate with strong lexical evidence. This restores the known-good relation/effect context while keeping a generic path for compact CSV/HTML/TXT/DOCX records.
- The global CrossEncoder threshold remains 0.55. No model, embedding, reranker, chunk schema, or requirements change is included.
- KB Health now distinguishes intentional minimum-length skips from genuine indexing failures. A file explicitly skipped as `too short [...]` remains visible in the report but is non-blocking; missing files, zero-chunk indexed files, and other skip reasons still fail the health gate.
- `Run_General_Corpus_Gate.bat` can therefore continue past a benign short-file skip and reach the certified 78-turn baseline plus runtime unseen-document add/query/remove proof.

Local regression before packaging: 345/345 tests pass with warnings treated as errors.

## v6.4.15.1 General Corpus Validation Gate

v6.4.15.1 keeps the v6.4.15 production retrieval architecture unchanged and adds only validation tooling for the pre-low-spec gate. `Run_General_Corpus_Gate.bat` performs the read-only KB health check, the exact certified 78-turn Cycle-1 regression run, and a self-cleaning runtime unseen-document proof. The unseen proof creates a TXT file with randomized facts under `data/all_documents`, incrementally indexes it, asks real AnswerService questions, requires the temporary source and values to be grounded, then deletes the file and incrementally removes it. A gate PASS therefore proves that an arbitrary newly added supported document can enter and leave the active knowledge base without production code changes.

No model, embedding, reranker, threshold, chunk/index schema, requirements, or production retrieval behavior changes are included.

## v6.4.15 General Corpus Retrieval Architecture Recovery

v6.4.15 is based strictly on the Windows/Ollama-certified v6.4.14.2.4
checkpoint (78/78 PASS). The rejected v6.4.14.2.5 manual-recovery branch is
not used as a base. This release re-centers DocuBot on its product contract:
`data/all_documents` is the knowledge base, and any supported file that is
successfully indexed must compete for retrieval without topic-, filename-, or
test-question-specific routing.

General corpus retrieval recovery:

- Hybrid merge now retains independent BM25/vector ranks and scores instead of
  collapsing all retrieval evidence into one opaque combined value.
- A lightweight corpus-wide lexical-coverage channel can surface terse exact
  records that ordinary top-K retrieval may omit when very large documents
  dominate the candidate set. It reuses already-loaded BM25 records and adds no
  model, vector index, or network dependency.
- The reranker input pool now preserves the ordinary ranked evidence first and
  uses high-confidence lexical coverage only as a bounded rescue path for a
  compact source that would otherwise be omitted. The pool size remains capped,
  preserving low-spec behavior.
- Strong lexical proof is an independent evidence path for compact exact source
  facts. The global reranker confidence threshold remains 0.55; the threshold
  is not lowered or bypassed for weak evidence.
- Generic list/completeness questions no longer reuse biography/relationship
  scoring. Relationship-specific ranking remains limited to actual relationship
  intent, while arbitrary CSV/table/list/procedure questions use domain-neutral
  evidence scoring.

General temporal and answer-safety recovery:

- Two-event temporal handling now recognizes generic `compare`, `which came
  first`, and chronological-order requests, not only one historical wording.
- Date/event binding supports both `October 20, 1944` and `20 October 1944`
  layouts and keeps the certified local-date isolation guard from v6.4.14.2.4.
- Equivalent event morphology is canonicalized generically (for example
  execution/executed, establishment/founded, capture/captured,
  surrender/surrendered, return/returned).
- Non-substantive generation artifacts such as a lone Markdown `>` are rejected.
  A compact source sentence can be recovered only from already accepted company
  context with strong lexical/value evidence; otherwise the normal grounded
  fallback is used.

Knowledge-base health visibility:

- `Run_KB_Health_Check.bat` creates a read-only coverage report under
  `logs/kb_health/`.
- It reports files discovered under `data/all_documents`, supported/unsupported
  types, manifest coverage, validation-skipped files, active Chroma chunk counts
  per file, zero-chunk files, and overall KB health.
- The health check does not load embeddings, reranker, or Ollama and does not
  rebuild or mutate the knowledge base.

QA evaluator stability:

- The R-012 robustness assertion now accepts semantically equivalent Tagalog
  country names (`Estados Unidos`, `Espanya` / `Kaharian ng Espanya`) so a
  grounded localized answer is not mislabeled as a product failure. This is an
  evaluator-only change and does not alter product answers or retrieval.

No model, embedding model, reranker model, global 0.55 threshold, chunk/index
schema, or requirements change is included. Existing `storage` should be kept;
no `pip install` and no forced full Knowledge Base rebuild are required.

Before lower-spec PC profiling:

1. Run `Run_KB_Health_Check.bat` and confirm HEALTHY.
2. Run `Run_Robustness_QA_Baseline_Compare.bat` and require 78/78 PASS.
3. Run the fresh/manual general-corpus questions and save the newest evidence log.
4. Only after those gates are green, copy the same application + existing storage
   to the lower-spec PC for cold/warm latency and memory validation.

## v6.4.14.2.4 Deterministic Temporal Finalization After Retrieval Retry

v6.4.14.2.4 is based strictly on v6.4.14.2.3. It preserves the accepted
v6.4.12 UI, hybrid model routing, reranker, 0.55 confidence threshold,
structured Example/code fidelity, source metadata, index schema, and all current
Cycle-1 passing behavior. This checkpoint addresses only the remaining R-015
routing/finalization defect observed in the actual Windows/Ollama Cycle-1 run.

Deterministic temporal finalization:

- Explicit two-event before/after comparisons are finalized before model routing
  whenever both requested event/date pairs are already grounded in accepted
  company context. This applies equally to initial retrieval and to the existing
  multilingual English-only retry path.
- Date/event binding now chooses a genuinely local structural unit. Compact prose
  binds within its own sentence, while wrapped PDF/table values use a backward
  local window that cannot cross a neighboring full date.
- This closes the observed table/prose boundary case where an `Effective` date
  immediately before a sentence about a different signing date could inherit the
  next event's `Treaty ... signed` words and make deterministic comparison
  ambiguous.
- If both event/date bindings are grounded, LLM generation is bypassed. The
  deterministic answer states the earlier and later dates directly; if the
  binding is incomplete or ambiguous, the existing grounded model path remains
  available instead of guessing.
- Cross-document deterministic temporal comparisons preserve one accepted source
  for each answer date. Ordinary answers keep the long-standing single-source
  display behavior.

Regression coverage:

- Added Tagalog, English, and mixed-language before/after tests using the same
  structural shape as the Windows retry evidence, including the adjacent
  `Effective April 11, 1899` distractor.
- Added a direct retry-path integration test that starts with empty bilingual
  retrieval, succeeds on the English-only retry, proves that LLM generation is
  not called, and verifies both grounding sources are returned.
- Existing temporal, structured-detail, source-metadata, Example/code-fidelity,
  and robustness suites remain unchanged except for package-version labels.

No model, embedding model, reranker, global threshold, chunk/index schema, or
requirements change is included. No `pip install` or Knowledge Base rebuild is
required; keep the existing `storage` directory.

After installing this checkpoint, run `Run_Robustness_QA_Baseline_Compare.bat`
with the unchanged Cycle 1 / seed 1789101877 baseline. Cycle 1 is not certified
until the real Windows/Ollama run reports a genuine 78/78 with correct R-015.

## v6.4.14.2.3 Tagalog Temporal Comparison Binding + Structured Example Fidelity

v6.4.14.2.3 is based strictly on v6.4.14.2.2. It preserves the accepted
v6.4.12 UI, hybrid model routing, reranker, 0.55 confidence threshold,
structured-detail absence messaging, source metadata, and the hardened QA
evaluator while closing the final Cycle-1 temporal retrieval gap and improving
source-faithful presentation of explicit Example/Examples blocks.

Tagalog temporal comparison binding:

- Tagalog comparison forms such as `nauna ba ... kaysa ...`, `mas nauna ba ...
  kaysa ...`, and `alin ang nauna/sumunod: ... o ...` are parsed into two clean
  event facets before retrieval.
- Whole-query reranking now preserves the strongest above-threshold candidate for
  every explicit compound facet. A second event can no longer be crowded out by
  several high-ranking chunks for the first event.
- The existing 0.55 reranker threshold is unchanged. Facet preservation never
  promotes a below-threshold candidate.
- Explicit two-event temporal comparisons require above-threshold retrieval
  coverage for both facets. If one side is not grounded, DocuBot falls back
  instead of returning one isolated date.
- Temporal deterministic answering remains strict: both requested events must be
  locally bound to their own grounded dates before an earlier/later answer is
  produced.
- Temporal comparison detection is also applied defensively before model routing,
  preventing a valid two-event comparison from silently falling through to a
  single-date LLM answer.

Structured Example fidelity:

- Direct Example/Examples requests preserve the source block's line order and
  meaningful line breaks instead of flattening everything into one paragraph.
- Code-like source runs are rendered in fenced C code blocks while source prose
  remains in separate readable paragraphs.
- Compliant/non-compliant source comments, labels, statements, braces, and code
  lines are retained from the extracted document; no new explanatory facts are
  invented.
- A source `Examples` heading remains plural in the answer, while a singular
  `Example` heading remains singular.
- Existing `See also` and neighboring structured blocks are not absorbed into the
  requested Example block.
- If no explicit Example/Examples block exists, the accepted specific English
  absence message remains unchanged.
- Final output-wrapper cleanup now removes only a true whole-answer markdown/text
  wrapper and preserves internal fenced code blocks.

QA / compatibility:

- A dedicated v6.4.14.2.3 regression suite covers the Windows R-015 failure
  pattern, additional Tagalog comparison forms, above-threshold facet coverage,
  below-threshold non-promotion, actual MISRA code examples, prose-plus-code
  separation, missing Example behavior, and internal code-fence preservation.
- No model, embedding model, reranker, global threshold, chunk/index schema, or
  requirements change is included.
- No `pip install` or Knowledge Base rebuild is required; keep the existing
  `storage` directory.

## v6.4.14.2.2 Temporal Comparison Correctness + QA Assertion Hardening

v6.4.14.2.2 is based strictly on v6.4.14.2.1. It preserves the accepted
v6.4.12 UI, hybrid model routing, reranker, 0.55 confidence threshold,
structured-detail behavior, source metadata, and all current deterministic
fast paths while closing the false-PASS temporal comparison found in the
Windows/Ollama 78-turn evidence run.

Temporal comparison correctness:

- Two-event date comparisons now bind each date to a local event window inside
  the same retrieved document. Event words cannot leak across retrieved-document
  boundaries or across a broad timeline and make an unrelated date look valid.
- Explicit month/year cues in a comparison facet are hard constraints. A query
  that explicitly says `June 1898` cannot be deterministically bound to an
  August 1898 or 1964 date.
- Generic multilingual event normalization is retained for relation families
  such as declaration/proclamation and signing/signed, including Tagalog cues.
- `before or after`, `which came later/first`, and Tagalog `nauna/sumunod ...
  kaysa ...` forms are all treated as two-event comparison intents.
- If both event/date bindings are not sufficiently grounded, the deterministic
  comparison path refuses to guess.

QA assertion hardening:

- R-015 now requires the actual grounded dates for both compared events and the
  correct earlier/later relation; a response can no longer pass merely because
  it contains `1898` and a temporal keyword.
- The exact false-PASS evidence from v6.4.14.2.1 (`August 13, 1898` versus
  `August 4, 1964`) is preserved as a regression test and must fail the QA
  evaluator.
- Nearby date assertions were reviewed so exact-date cases continue to require
  their grounded dates rather than broad year-only matching.

A fresh manual-validation guide is included separately and is built from the
actual files in `data/all_documents` (CSV, HTML, TXT, policy, historical PDFs,
and MISRA). Those questions are intentionally not executed by the automated
unit/regression suite so they remain a human reality-check after the Windows
78-turn gate passes.

No model, embedding model, reranker, global threshold, chunk/index schema, or
requirements change is included. No `pip install` or Knowledge Base rebuild is
required; keep the existing `storage` directory.

## v6.4.14.2.1 Final Cycle-1 100% Recovery + Source Metadata + Explicit Missing-Detail Messaging

v6.4.14.2.1 is based strictly on v6.4.14.2. It keeps the accepted v6.4.12
sidebar/UI, current hybrid models, reranker, 0.55 confidence threshold, chunk/index
schema, and low-spec/lazy-loading architecture unchanged while addressing the four
remaining Cycle-1 failures and the final structured-source quality gaps.

Structured missing-detail behavior:

- If an exact Rule/Directive/Section/Article/Chapter/Part exists but a requested
  supported subfield is absent, DocuBot now returns a specific English absence
  message such as `No Amplification section is provided for Rule 1.2 in the
  available company knowledge.` instead of the generic company-KB fallback.
- Supported named details include Category, Analysis, Applies to, Amplification,
  Rationale, Example(s), Exception(s), and See also. Extraction remains strictly
  label-preserving: inline phrases such as `for example` are not promoted into an
  Example block, and nearby Rationale text cannot substitute for Amplification.
- If the structured reference itself does not exist, the standard exact fallback
  remains `Information not found in company knowledge base.`
- Verified missing-detail answers retain the accepted source, Page/Pages metadata,
  and exact structured reference so the UI/evidence output can show where the
  absence was checked.

Final Cycle-1 / answer-quality recovery:

- Windows-safe `#UXXXX` filename markers are decoded only for topic matching and
  display, allowing names such as Jose/José Rizal to match without changing the
  physical private-document path.
- Treaty `it` follow-ups and Tagalog `nito` founder follow-ups retain the latest
  explicit object even after query normalization.
- Before/after date questions use two-event/date binding; generic numeric extraction
  is blocked from treating unrelated years or nearby fragments as a comparison.
- Short Tagalog entity descriptions can use a clean grounded identity overview
  instead of drifting into incidental document details.
- Exact Section subtopic selection prefers explanatory prose over presentation/schema
  rows for concepts such as decidability. An actual MISRA PDF regression protects
  this behavior without hardcoding the answer text into production logic.
- QA evidence source normalization now preserves `page_start`, `page_end`, and
  structured `reference` fields rather than dropping them after answer generation.

Validation in this package is local/static only. The real Windows/Ollama 78-turn
Cycle-1 run is still required before claiming 78/78. No `pip install` or Knowledge
Base rebuild is required; keep the existing `storage` directory.

## v6.4.14.2 Structured Completeness + Final Cycle-1 Recovery

v6.4.14.2 is based strictly on v6.4.14.1. It preserves the accepted v6.4.12
sidebar/UI behavior, the v6.4.13 latency architecture, and the v6.4.14/v6.4.14.1
robustness recovery while closing the remaining structured-document completeness
and Cycle-1 issues identified by the Windows/Ollama 71/78 evidence run.

Structured completeness / label-preserving changes:

- `Exception` / `Exceptions` are now first-class exact Rule/Directive detail blocks,
  alongside Category, Analysis, Applies to, Amplification, Rationale, Example(s),
  and See also.
- Exact structured identifiers now support `Article`, `Chapter`, and `Part` aliases
  in addition to Rule, Dir/Directive, and Section. Section-like aliases reuse the
  existing section metadata/index path and do not require a KB rebuild.
- Structured detail extraction is strictly label-preserving. A request for
  Amplification, Rationale, Example(s), Exception(s), or See also can only return
  that explicit labeled source block. If the requested block is absent, DocuBot
  returns the standard company-knowledge fallback instead of relabeling nearby
  rationale prose or inline phrases such as "for example".
- Full Rule/Directive explanations can include explicitly labeled Exception(s) and
  See also content when present.

Final Cycle-1 recovery changes:

- Section topic extraction now supports one-line and two-line subsection headings,
  preventing Section 6 topic/overview questions from collapsing to generic terms.
- Section overview requests return the exact structured heading plus its opening
  supported description instead of being treated as topic-list requests.
- Rule-vs-Directive distinction questions are routed to exact Section detail and
  retain enough local evidence to cover both sides of the distinction.
- Terse structured follow-ups such as `What does it say about decidability?` now
  retain the latest exact Section/Article/Chapter/Part reference when the wording
  clearly indicates a structured attribute/detail follow-up.
- Tagalog object-reference recovery now handles a compact prior named subject in
  chains such as `Ano ang Katipunan sa maikling paliwanag?` ->
  `Sino ang isa sa mga nagtatag nito?` without promoting the whole explanatory
  phrase to the topic.
- Strict identity fast-path definition matching now tolerates bounded middle-name
  expansion between the requested first/last name anchors, which protects profile
  retrieval for sources whose lead sentence uses a longer legal name.
- Explicit before/after temporal comparisons are treated as two-event retrieval
  intents (including Tagalog `nauna ... kaysa ...`) so both compared events can
  survive candidate collection and deterministic answer selection.

Validation in the packaged codebase must still be followed by the real Windows/Ollama
Cycle-1 run before claiming 78/78. No model, embedding model, reranker, global 0.55
confidence threshold, chunk/index schema, or requirements change is included. No
`pip install` or Knowledge Base rebuild is required. Keep the existing `storage`
directory.

## v6.4.14.1 Structured Detail & Citation Enhancement + Robustness Cleanup

v6.4.14.1 is based strictly on v6.4.14. It preserves the accepted v6.4.12
sidebar/UI behavior and the v6.4.13 latency architecture while adding the
planned structured-detail/citation enhancement and focused fixes for the
remaining Cycle-1 robustness failures.

Structured detail / citation changes:

- Exact Rule/Directive questions can now request **Amplification**, **Rationale**,
  or **Example/Examples** directly from the already accepted structured block.
- Full exact Rule/Directive explanations label supported Amplification and
  Rationale content and include supported Example/Examples when present.
- Source objects now carry `page_start`, `page_end`, and the best available
  exact structured reference. The Streamlit source chip displays Page / Pages
  and Rule/Directive/Section reference when that metadata exists.
- QA evidence logging now understands `page_start` / `page_end` rather than
  relying only on legacy page-number keys.

Focused robustness cleanup:

- Identity BM25 proof no longer treats incidental clauses such as `X was able`
  as a biographical definition. Deterministic identity extraction also requires
  a true definitional predicate plus an identity signal.
- Rule/Directive follow-ups such as `Which C versions does it cover?` use the
  resolved structured reference and deterministically return the complete
  `Applies to` value when present.
- Section topic/overview questions can return concise child subsection headings
  instead of dumping repeated body/reference terms. Named Section follow-ups
  such as decidability remain bound to targeted structured detail retrieval.
- Compact two-value numeric comparisons can be answered deterministically when
  both labels and values are explicit in accepted context; explicit before/after
  date comparisons can likewise preserve the relation instead of returning only
  one date.
- Generic Tagalog founder follow-ups and annual allotment phrasing receive
  retrieval-friendly canonical forms without hardcoding a corpus answer.
- Robustness QA accepts compact inline enumerations and grounded wording aliases
  where the previous evaluator produced clear false negatives.

No model, embedding, reranker, confidence threshold, chunk/index schema, or
requirements change is included. No `pip install` or Knowledge Base rebuild is
required. Keep the existing `storage` directory.

After installation, rerun `Run_Robustness_QA_Baseline_Compare.bat` with the same
Cycle 1 / seed 1789101877. The package is not considered 78/78 Windows/Ollama
validated until that real run is supplied and reviewed.

## v6.4.14 Robustness Recovery + QA Evaluator Hardening

v6.4.14 is based on v6.4.13.1 and responds directly to the first 78-turn
randomized robustness run (Cycle 1 / seed 1789101877), which exposed natural
paraphrase, Tagalog, multi-turn, and answer-focus weaknesses that were not
visible in the older familiar regression prompts.

Production recovery changes are domain-neutral:

- Profile/identity paraphrases such as concise profile / identify-in-documents
  are normalized into a stable identity retrieval intent so strict identity
  grounding can still select introductory source evidence.
- Eligibility/approval paraphrases now cover natural forms such as "who
  qualifies", "covered by", "who must approve", and "whose approval is
  required" without hardcoding a policy or answer.
- Tagalog relation normalization now covers additional generic vocabulary for
  approval/permission, annual quantities, purpose/goal, position, territories,
  declaration/signing/date relations, and common follow-up wording while the
  original-language query remains searched in parallel.
- Conversation recovery now preserves object/possessive references such as
  it/this/these and ito/iyan/iyon/nito/niyan, broadens person-referent recovery
  for profile-style introductions, and keeps terse Rule/Directive/Section
  follow-ups bound to the latest exact structured identifier.
- Strict BM25 direct-relation proof now recognizes purpose and eligibility
  relations in addition to the earlier v6.4.13 factual relations. Weak or
  ambiguous matches still fall back to the normal semantic/reranker pipeline.
- Answer-focus detection is stronger for quantities, approval, eligibility,
  purpose, comparisons, requested subject matter, and target-specific effects.
  This reduces cases where a true nearby fact answers the wrong facet.
- Exact Section questions that name a specific subtopic can use a deterministic
  source-only local passage extractor instead of returning the generic Section
  introduction. The extractor never searches outside the already accepted
  exact Section block.
- Multi-value list detection/presentation now handles plain multi-line values
  and compact comma-separated enumerations more consistently.

QA evaluator hardening:

- Conditions that are part of the question but do not need to be repeated in
  the answer are no longer treated as mandatory output text for R-050.
- Semantically valid independence wording is accepted for the Mabini
  significance cases.
- Section-overview QA now detects runaway/excessive list output so a keyword
  match cannot hide an obviously over-expanded answer.
- Expected-answer metadata remains post-answer-only and is never supplied to
  AnswerService, retrieval, prompts, or model routing.

For apples-to-apples comparison with the v6.4.13.1 evidence, run
`Run_Robustness_QA_Baseline_Compare.bat`. It uses the same full suite,
Cycle 1, and seed 1789101877. Use the original `Run_Robustness_QA.bat` for a
fresh randomized run after the repeatable baseline comparison is reviewed.

No Knowledge Base rebuild or `pip install` is required for this code update.
Keep the existing `storage` directory.

## v6.4.13.1 Automated Robustness QA Runner

This QA-only checkpoint sits on top of v6.4.13 and does not change production
retrieval, answer generation, model routing, UI, indexing, embeddings, reranker,
or thresholds. It adds a no-manual-typing robustness runner with 50 standalone
question families (multiple paraphrases each), seven real multi-turn chains,
randomized execution, post-answer-only automated checks, latency/route/model
summary capture, CSV/JSON/TXT reports, and a Windows double-click launcher
`Run_Robustness_QA.bat`. Expected-answer metadata is never supplied to
`AnswerService.ask()`. Use `--cycle 2` / `--cycle 3` to rotate question wording.

## v6.4.13 Latency Phase 2 - Direct Relation Fast Path + Low-Spec Ollama Policy

v6.4.13 resumes the accepted v6.4.12 UI checkpoint and changes only
latency/runtime behavior. The sidebar accepted in v6.4.12 is preserved. Models,
embedding model, reranker model, global `0.55` confidence threshold, chunking,
index schema, and company-only grounding rules remain unchanged.

- A strict **BM25 direct-relation fast path** now covers narrow factual
  position/role, organization/founder, approver, quantity, and date/time
  questions when one compact company chunk explicitly contains the requested
  anchor and relation/value. Safe matches skip Chroma embedding and CrossEncoder
  loading entirely. Explanations, lists, compounds, structured references, weak
  matches, and close cross-source conflicts automatically fall back to the
  existing vector + reranker pipeline.
- The direct-relation gate requires local anchor coverage plus relation-specific
  lexical/value evidence. An absent question such as an employee parking
  reimbursement limit cannot qualify merely because unrelated documents contain
  generic words such as `employee`, `reimbursement`, or `limit`.
- The frequently used fast model now uses a **4096-token context window** by
  default instead of the previous universal 8192-token window. v6.4.7+ already
  compacts simple fast-model generation context, so this reduces avoidable
  runtime memory pressure while leaving Qwen complex work at 8192 tokens. Both
  values remain configurable by environment variables.
- Ollama residency is now model-specific and configurable. The fast model
  defaults to `15m` keep-alive, while the larger complex model defaults to `2m`
  so constrained PCs can release its memory sooner. `DOCUBOT_OLLAMA_KEEP_ALIVE`
  remains available as a single diagnostic/deployment override.
- QA evidence for every LLM call now records the routed model's `context_window`
  and `keep_alive` policy, making the next Windows evidence run suitable for
  measuring model-residency/swap behavior rather than inferring it indirectly.

Optional runtime overrides:

```powershell
$env:DOCUBOT_FAST_LLM_CONTEXT_WINDOW = "4096"
$env:DOCUBOT_COMPLEX_LLM_CONTEXT_WINDOW = "8192"
$env:DOCUBOT_FAST_LLM_KEEP_ALIVE = "15m"
$env:DOCUBOT_COMPLEX_LLM_KEEP_ALIVE = "2m"
```

No `pip install` or Knowledge Base rebuild is required from a healthy v6.4.12
deployment. Keep the existing `storage` directory.

## v6.4.12 Sidebar Bottom Visibility Only

UI-only correction based strictly on v6.4.11. The accepted scrollbar position
and recent-chat card geometry are preserved exactly. The only layout change is
a 32px larger bottom safe reserve on the Recent Chats scroll viewport, so the
last card can be scrolled fully above the visible sidebar bottom instead of
being clipped. No RAG, retrieval, model, embedding, reranker, threshold,
indexing, or latency logic is changed.

# Company Knowledge Assistant

Private Offline Company Knowledge Search

## Features

- PDF
- DOCX
- XLSX
- PPTX
- TXT
- CSV
- HTML
- XML
- MD

## Stack

- Streamlit
- LlamaIndex
- Ollama
- ChromaDB
- BM25
- BAAI/bge-reranker-base

## Knowledge Base Operations

Normal document maintenance uses one smart operation:

```powershell
python -m scripts.smart_build
```

The Streamlit UI exposes only **Update Knowledge Base**. The backend decides
whether the operation can be incremental or whether a safe full rebuild is
required because the vector/index identity is incompatible or the active index
is missing/unusable. A full-rebuild confirmation appears only when that case is
detected. The rollback-capable full-rebuild engine remains available internally
and through the maintenance script:

```powershell
python -m scripts.rebuild_index
```

## Run

```powershell
streamlit run app.py
```




## v6.4.9 Sidebar Fixed Header + Scrollable Recent Chats

v6.4.9 is a focused UI-only patch on top of v6.4.8. It keeps the company logo,
DocuBot title/subtitle/version, and New Chat control stationary in the sidebar.
Only the Recent Chats card list scrolls when conversation history grows. The
main chat layout, RAG pipeline, models, embeddings, reranker, thresholds,
indexing, and v6.4.8 grounding/latency behavior are unchanged.

No `pip install` or Knowledge Base rebuild is required from a healthy v6.4.8
installation.

## v6.4.8 BM25 Fast-Path Grounding Hardening + Latency Phase 2 Continuation

v6.4.8 is a focused correctness hardening on top of v6.4.7 while preserving
the latency gains that were measured on the user's Windows evidence run. Models,
embeddings, reranker configuration, the global `0.55` threshold, chunking, and
index schema remain unchanged.

- The BM25-first identity gate is now **subject-bound**. The exact requested
  subject must appear near the opening of the same candidate chunk and must
  directly participate in an introductory definition/profile relation. A
  same-file narrative chunk can no longer qualify merely because it mentions
  the subject and contains generic words such as `was`, `founding`, `leader`,
  or `organization` elsewhere.
- Generic narrative/reference headings such as References, Bibliography, Notes,
  Citations, External Links, Marriages, Personal Life, Commemoration, Legacy,
  Awards, Education, and Early Life are explicitly ineligible for the identity
  fast path. If no safe introductory definition is found, retrieval falls back
  to the normal vector + reranker pipeline.
- The fast-path event now records the exact selected `chunk_id`, requested
  target, definition score, and safety score so Windows evidence can prove that
  the chunk used for the answer is the chunk that actually passed the gate.
- When the strict BM25 identity fast path succeeds, DocuBot now extracts the
  user-facing identity sentence **deterministically from the accepted source
  context**. It does not ask Llama/Qwen to rewrite that sentence. This prevents
  outside-knowledge expansion and also removes first-model generation cost on
  that safe path, which is especially useful on lower-spec CPU/RAM machines.
- The existing lazy vector/reranker loading and generation-only fast-context
  compaction remain unchanged. Semantic retrieval still loads only when a
  question cannot be answered by an exact structured or strictly proven BM25
  fast path.

The v6.4.7 Windows evidence showed that the first identity lookup reduced
retrieval to about `0.07s` by deferring semantic model load, but also exposed a
Bonifacio grounding defect because the selected chunk was a Marriages narrative
section. v6.4.8 closes that defect without lowering the confidence threshold or
loosening semantic retrieval.

No `pip install` or Knowledge Base rebuild is required from a healthy v6.4.7
deployment. Do not delete `storage`. v6.3.4 remains the locked safe fallback
until Windows/Streamlit/Ollama validation is complete.


## v6.4.7 Latency Optimization Phase 2 - BM25 First + Context Reduction

v6.4.7 continues the low-risk latency work on top of v6.4.6 while keeping
retrieval quality guardrails, models, embeddings, reranker configuration, the
global `0.55` threshold, chunking, and index schema unchanged. History, MISRA,
and leave-policy files remain regression corpora only; the optimizations are
domain-neutral.

- A strict **BM25-first identity/overview fast path** can now answer a first
  `Who is X?`-style lookup without loading Chroma embeddings or the CrossEncoder
  when BM25 already found a clean introductory chunk whose source topic matches
  the requested subject and whose text contains an explicit definition/profile.
  Reference/citation noise, weak source-topic matches, compounds, list requests,
  and ambiguous cases automatically fall back to the existing vector + reranker
  pipeline.
- Simple fast-model prompts can use a compact generation-only copy of accepted
  context. The compactor keeps the source opening plus the densest relation
  window and caps the initial fast-model payload conservatively. The original
  full accepted context is retained for deterministic safety, verification, date
  relation handling, citations, and returned chunks. Compound/list/explanation
  paths are not compacted.
- Narrow direct organization questions now recognize `help found` / `founded`
  wording in addition to `co-found`. When accepted company context explicitly
  states that relationship, the existing grounded relation guard can preserve
  the exact organization without an unnecessary second verifier call.
- Short quantity, time, location, and approver drafts can skip focus verification
  only when the returned value is explicitly proven in a local relation window.
  For compact label/value sources, label pairing is required; a neighboring value
  from another field cannot qualify merely because it appears in the same chunk.
- Canonical English semantic targets are also used to recognize direct
  quantity/approver/time/location intent for multilingual questions. This keeps
  the user's language for the answer while avoiding an unnecessary generic
  identity/focus verifier on clearly translated direct relations.
- QA evidence records `BM25 FIRST PASS / IDENTITY FAST PATH` and
  `FAST CONTEXT COMPACTED` events so the next Windows run can measure how much of
  the first-use and warm-prompt cost was removed.

No `pip install` or Knowledge Base rebuild is required from a healthy v6.4.6
deployment. Do not delete `storage`. v6.3.4 remains the locked safe fallback
until Windows/Streamlit/Ollama validation is complete.


## v6.4.6 Verifier Cleanup + Latency Optimization Phase 2 Start

v6.4.6 is a focused follow-up on top of v6.4.5. It closes the remaining
answer-focus verifier presentation regression and begins low-risk startup
latency work without changing models, embeddings, reranker scoring, thresholds,
chunking, or retrieval semantics.

- The answer-focus verifier now strips an exact echoed user/resolved query line
  before final presentation. Internal routing text such as
  `why was Emilio Aguinaldo important` can no longer be prepended to an
  otherwise grounded answer.
- Importance/significance follow-ups can skip the second focus-verifier LLM call
  when the first grounded draft already names the resolved subject and provides
  a complete supported significance/role statement. This narrow optimization
  does not apply to ordinary causal WHY questions, which keep the verifier path.
- Invalid standalone `Yes`/`No` prefixes are removed before focus verification
  for non-YES/NO intents so they cannot distort verifier behavior.
- The LlamaIndex Ollama adapter is imported only when generation is actually
  requested. Creating a new Streamlit process or answering a deterministic
  structured query no longer needs to import that adapter up front.
- `CompanyRetriever` now loads BM25 immediately but defers Chroma/embedding and
  CrossEncoder reranker construction until semantic retrieval actually needs
  them. Exact Rule/Directive/Section lookup can therefore remain on the
  structured BM25 metadata path without loading unused ML models.
- Startup-only Knowledge Base planning no longer imports the update-only
  ingestion/embedding/full-index pipeline. Those heavier modules are imported
  only when an actual Knowledge Base update/rebuild operation needs them.
- QA evidence now records retriever-wrapper initialization time plus separate
  cold-load durations for the lazy vector/embedding and reranker components,
  making cold vs warm latency easier to diagnose on lower-spec PCs.

No Knowledge Base rebuild is required from a healthy v6.4.5 deployment. Do not
delete `storage`. v6.3.4 remains the locked safe fallback until local
Windows/Streamlit/Ollama evidence validates this checkpoint.


## v6.4.5 Recovery Closure

v6.4.5 is a focused recovery-closure patch on top of v6.4.4. It remains
domain-neutral: history, MISRA, and leave-policy documents are regression
corpora only, while runtime behavior is driven by whatever supported files are
present in the configured folder/storage.

- Short person-pronoun relation follow-ups such as `What position did he hold?`
  can no longer be incorrectly treated as a brand-new topic before typed person
  reference recovery runs. The bypass is limited to clear pronoun-follow-up
  grammar so an explicitly named current subject still remains authoritative.
- Deterministic Tagalog relation normalization now covers additional generic
  quantity/provision/approval vocabulary (`araw`, `ibinibigay`, common
  `nag-aapruba`/approval forms, and `kailangan`) and removes a few residual
  grammar particles from the supplemental canonical query. No document, person,
  policy, or historical fact is hardcoded.
- The original-language query continues to be searched in parallel. No model,
  embedding, reranker, global confidence threshold, chunking, index schema, or
  Knowledge Base build behavior is changed.

No `pip install` or Knowledge Base rebuild is required from a healthy v6.4.4
deployment. Do not delete `storage`. v6.3.4 remains the locked safe fallback
until Windows/Streamlit/Ollama evidence validates this checkpoint.


## v6.4.4 Context + Multilingual + Grounding Recovery

v6.4.4 is a focused regression-recovery patch on top of v6.4.3 using the
latest Windows evidence log. It does not introduce document-specific answer
logic. Philippine-history, MISRA, and leave-policy files remain test corpus
only; retrieval behavior stays domain-neutral and is driven by whatever
supported files are present in the configured folder/storage.

- Person-pronoun follow-ups now prefer the latest person explicitly introduced
  by the user (for example through `Who is X?` / `Who was X?` / `Sino si X?`)
  instead of blindly inheriting a broad retrieval-derived `current_topic`. This
  fixes multi-turn cases where a role/position lookup changes the broad topic
  before a later `he`/`she` follow-up. Existing object/structured pronoun logic
  remains separate.
- Multilingual query detection now gives strong Tagalog interrogatives priority
  over incidental English function words inside names/titles. A query such as
  `Kailan nilagdaan ang Treaty of Paris ng 1898?` is therefore no longer
  misclassified as English merely because the title contains `of`.
- Common high-confidence Tagalog question/relation cues receive a deterministic
  supplemental canonical retrieval query before any LLM rewrite. The untouched
  original-language query is still searched in parallel, so names, identifiers,
  numbers, and technical terms remain available. Other languages/ambiguous cases
  retain the existing local-model multilingual fallback.
- Compound-facet anchoring is narrower. A complete second clause such as
  `who must approve a leave request` no longer inherits unrelated residue from
  the first clause merely because it is short; pronoun-dependent fragments
  still receive the shared subject anchor.
- Compound verification is now validation/minimal-repair oriented. The
  deterministic coverage gate recognizes imperative facets such as
  `explain it` and `explain why ...`, allowing already-complete grounded drafts
  to skip a second Qwen call. If verification is still needed, an additional
  guard rejects large answer expansion or weakly grounded new content and keeps
  the original grounded draft.

No model, embedding, reranker, global `0.55` threshold, chunking, index schema,
requirements, or Knowledge Base build behavior is changed. No `pip install` or
Knowledge Base rebuild is required from a healthy v6.4.3 deployment. Do not
delete `storage`. v6.3.4 remains the locked safe fallback until this package is
validated on the target Windows/Streamlit/Ollama environment.

## v6.4.3 Mixed Intent + Multi-Turn Context Recovery

v6.4.3 is a focused follow-up to v6.4.2 after Windows evidence showed that
exact structured retrieval could answer the `what` part of a same-turn request
but ignore a later instruction such as `and explain it as well`. The same QA
run also exposed a compact leave-policy compound question that retrieved the
correct policy block but still returned the strict fallback.

- Same-turn request detection now recognizes a second explicit instruction as
  well as a second interrogative. Examples include `What is Rule 13.5 and
  explain it as well` and `What is X? Explain it`. This logic is shared by
  answer-focus detection, compound facet retrieval, and compound answer checks.
- For one exact Rule/Directive/Section reference, explicit explanation intent
  has priority wherever it appears in the turn. Rule/Directive `what + explain`
  requests therefore keep the deterministic no-LLM path while returning the
  exact statement plus supported rationale/amplification. Section explanations
  continue to use the complex Qwen route.
- Terse multi-turn follow-ups now prefer the latest exact Rule/Directive/Section
  identifier over a broader document topic when the user says `Explain it`,
  `What category does it belong to?`, `What does it apply to?`, `the rationale?`,
  and similar structured-detail follow-ups. A conservative vocabulary guard
  prevents an explicitly named new subject from being overwritten by old
  structured context.
- Exact Rule/Directive `Category`, `Applies to`, and `Analysis` follow-ups can
  be answered deterministically from the accepted structured block, avoiding an
  unnecessary local-model call on lower-spec PCs.
- Compact label/value company sources can deterministically answer narrow
  multi-part quantity/approval/eligibility/entitlement questions only when every
  requested facet is explicitly matched in accepted context. This fixes the
  observed `How much sick leave ... and who must approve ...?` false fallback
  without lowering the retrieval threshold or relaxing grounding.

The v6.4.2 presentation/relation fixes, v6.4 latency instrumentation, v6.3.4
reranker numerical safety, v6.3.3 compound coverage, and all Knowledge Base
update protections remain unchanged. No model, embedding, reranker, threshold,
chunking, index-schema, requirements, or KB-build change is included. No
`pip install` or Knowledge Base rebuild is required from a healthy v6.4.2
deployment. Do not delete `storage`.

## v6.4.2 Answer Quality + Consistent List Presentation Recovery

v6.4.2 is a focused quality/presentation recovery patch on top of v6.4.1. It
uses the fresh Windows evidence to fix concrete wrong/fallback answers while
preserving the Phase 1 latency gains and all existing grounding safeguards.

- Multi-point explanations and explicit compound/list answers now receive a
  deterministic Markdown presentation pass after answer safety checks. Existing
  useful lists are preserved, simple one-fact answers remain concise, and a
  fake one-item numbered list is de-numbered. This adds no LLM call.
- Direct cause, organization/co-founder, position/role, and title questions get
  tighter answer-focus rules. When accepted company context contains an
  explicit lexical relation, a deterministic grounded relation guard can
  preserve that exact fact instead of accepting a nearby metadata value or an
  unnecessary fallback. Confirmed/conclusive causes take precedence over
  explicitly debunked rumors.
- Tagalog/non-English questions keep the user's original language for answer
  generation, while the already-created canonical English retrieval query is
  reused as the semantic relation target for retrieval intent, routing, and
  verification. This does not add another translation or LLM call.
- Identity/overview output removes PDF lead-block presentation noise such as
  duplicated names, IPA pronunciation text, and small footnote markers while
  preserving the supported source predicate.
- Common Tagalog definition forms such as `Ano ang ...?` now receive the same
  definition/detail focus used by equivalent English questions.

The embedding model, reranker, `0.55` threshold, chunking, index schema, hybrid
Llama/Qwen defaults, v6.4 latency instrumentation/verifier-skip gates, compound
facet retrieval, reranker numerical safety, and Knowledge Base update/build
semantics are unchanged. No `pip install` or Knowledge Base rebuild is required
from a healthy v6.4.1/v6.4 deployment. Do not delete `storage`.

## v6.4.1 Regression Recovery - Compound Intent + Tagalog Identity + Section Depth

v6.4.1 is a focused quality-recovery patch on top of v6.4 Phase 1. It
preserves the latency instrumentation and deterministic fast paths while fixing
three issues exposed by the fresh Windows validation questions.

- Compound eligibility/entitlement questions keep their original clause order
  through normalization. Single-intent canonical rewrites are skipped when a
  second explicit interrogative clause is present, so questions such as
  `What is X and who is eligible for it?` are not rearranged before retrieval.
- Common Tagalog identity forms such as `Sino si ...?` and `Sino ang ...?`
  normalize to the named subject and receive the same `IDENTITY OR OVERVIEW`
  safeguards as English `Who is ...?` questions.
- Exact Section explanation fallback now supplements title-only subsection
  output with a few source-grounded bullet details when the PDF layout stores
  explanatory bullets separately from subsection titles. Code-comment lines
  are not treated as content bullets.

The v6.4 direct Section no-LLM path, verifier-skip gates, latency logging,
compound-facet retrieval, reranker numerical safety, strict fallback, and all
knowledge-base/index settings remain unchanged. No `pip install` or Knowledge
Base rebuild is required from a healthy v6.4 deployment.

## v6.4 Answer Latency Optimization - Phase 1

v6.4 starts latency work from the Windows-validated v6.3.4 hybrid-routing
baseline without changing retrieval models, confidence thresholds, chunking,
index schema, or knowledge-base build semantics.

Phase 1 removes only redundant model work when a deterministic safety check can
prove that the extra call is unnecessary:

- Direct exact `Section N` lookups such as `What is Section 6?` now use the
  already-retrieved structured section title + opening description and bypass
  LLM generation. `Explain Section N` remains on the complex Qwen route.
- A concise one-sentence `IDENTITY OR OVERVIEW` draft may skip the second
  focus-verifier call only when it visibly contains the requested subject, an
  identity verb, and useful descriptive content. Richer or ambiguous drafts
  continue through the established verifier.
- Explicit compound questions may skip the second compound-verifier call only
  when every recognized facet passes a conservative deterministic coverage
  check. Ambiguous facet types keep the existing LLM verifier.
- QA evidence now records per-LLM-call latency plus a request latency profile,
  so Windows tests can separate retrieval time, generation time, and redundant
  verification cost.

The existing fallback, output-safety, structured exact lookup, compound facet
retrieval, reranker numerical-safety, and identity preservation guards remain
active. No `pip install` or Knowledge Base rebuild is required from a healthy
v6.3.4 deployment.

Every release from v6.4 onward keeps the fixed core regression checks and also
ships a rotated set of fresh manual questions so validation is not limited to
memorized/repeated prompts.

## v6.3.4 Reranker Numerical Safety + Identity Overview Preservation

v6.3.4 is a focused stability patch after Windows QA showed two related
non-compound identity anomalies while the v6.3.3 compound-facet fix itself was
working correctly. A rare CrossEncoder score was logged as `nan`, and the
identity verifier then reduced an otherwise useful `Who is ...?` answer to the
subject name only.

The reranker now validates every CrossEncoder score before sorting or applying
the confidence threshold. A non-finite score (`NaN` or `Inf`) is retried once
for that candidate at `batch_size=1`. If the retry becomes finite, the recovered
score is used normally. If it remains non-finite or the retry fails, DocuBot
assigns the finite score `0.0`, so the candidate is safely rejected by the
existing `0.55` confidence gate instead of entering unordered Python sorting.
QA evidence records a `RERANKER NUMERICAL SAFETY` event whenever this path is
used.

For `IDENTITY OR OVERVIEW` questions only, a second deterministic guard prevents
a verifier from collapsing a supported overview to a name-only answer. If the
verifier output is thin and the already-accepted company context contains a
direct definitional identity sentence, DocuBot returns that exact grounded
sentence after whitespace cleanup. It does not preserve unsupported draft
claims, does not use outside knowledge, and does nothing when the verifier
already returned a useful answer or the context lacks a direct identity fact.

No model, embedding, reranker model, retrieval threshold, chunking, index
schema, knowledge-base build, hybrid-routing policy, or requirements changes
are included. No `pip install` or Knowledge Base rebuild is required from a
healthy v6.3.3 deployment.

## v6.3.3 Compound Facet Retrieval Coverage

v6.3.3 is a focused retrieval-completeness patch after Windows validation
confirmed that the v6.3.2 structured index, exact Rule/Directive/Section paths,
and hybrid router were working again, but one explicit multi-part question could
still fall back even when evidence for both clauses existed in the same source.

For explicit compound questions, DocuBot now keeps the non-enriched resolved
question as the retrieval intent, splits only on a conjunction followed by a
second explicit interrogative, and performs small per-facet retrieval passes.
Pronoun-heavy later facets inherit a compact subject anchor from the first
facet (for example, `why he was significant` is searched with the subject from
`who Jose Rizal was`). Facet candidates are merged into the normal candidate
pool and still pass through the existing informative filters, CrossEncoder
reranker, `0.55` confidence gate, grounding rules, and compound-answer
verification.

Compound questions may also keep up to the normal final top-k chunks from one
source. This prevents a relevant third chunk from being discarded only because
two higher-ranked chunks came from the same document. Non-compound diversity
behavior is unchanged. No LLM, embedding, reranker, chunking, index schema,
knowledge-base build, or requirements changes are included.

## v6.3.2 Rebuild-Loop / Streamlit Disconnect Recovery

v6.3.2 is a focused follow-up to v6.3.1 after Windows logs showed that a
healthy structure-aware rebuild could be requested repeatedly. The integrity
probe had treated every semantic PDF section as if it required a numeric
`section_id`. Ordinary headings such as “Early life” are intentionally stored
as `section_type=section` without a numeric identifier, so the probe kept
classifying the freshly rebuilt index as damaged.

The integrity check now requires `section_id` only when the heading explicitly
looks like a numbered `Section N` identifier. Structure-aware semantic section
headings remain valid. After any repair full rebuild, Smart Update immediately
re-validates the committed index and reports an explicit error if it would
still request another full rebuild, preventing a hidden rebuild loop. No model,
embedding, reranker, retrieval-threshold, chunking, or index-schema changes are
included.

## v6.3.1 Structured-Index Recovery + Release Hygiene

v6.3.1 is a focused recovery patch after Windows validation exposed that the
previous update ZIPs could carry stale mutable `storage/` runtime state. Copying
those ZIPs with **Replace** could overwrite a deployment's working
structure-aware Chroma/BM25 index even though the application code itself had
not changed the retrieval implementation.

This patch makes normal update packages deployment-safe:

- Active `storage/chroma_db`, `storage/bm25`, `storage/metadata`, ingestion
  cache contents, legacy `storage/storage.zip`, and runtime logs are excluded
  from code-update ZIPs. Existing deployment data is therefore preserved when
  the package is copied over an installed project.
- `Update Knowledge Base` now performs a cheap structure-aware PDF metadata
  integrity probe. If tracked PDF vectors use legacy/generic metadata, are
  missing, or lack the required structural fields, the same smart update flow
  requests one safe staged full rebuild. There is still no permanent second
  rebuild button.
- The prepared-chunk cache schema is bumped so an integrity-repair rebuild
  cannot reuse stale pre-structure PDF chunks.
- Conversation resolution now gives an explicit current Rule/Directive/Section
  identifier or explicitly named current subject priority over the previous
  chat topic. This prevents cross-topic contamination such as replacing
  `he` in a Jose Rizal question with `Employee Leave Policy`.

No `pip install` is required. The embedding model, reranker, retrieval
threshold, chunk/index schema, and hybrid LLM routing policy are unchanged.

**Important for a machine that already ran the affected v6.3 package:** the
existing index may already have been overwritten. On first v6.3.1 startup,
DocuBot should detect that condition and show the conditional confirmation for
`Update Knowledge Base`. Approve it once so the existing safe staged rebuild
can restore structure-aware vectors. A healthy v6.2.1/v6.1 structure-aware
index does not need a rebuild solely because of the code update.

## v6.3 Hybrid LLM Routing + No-LLM Structured Fast Path

Normal DocuBot startup now uses both local Ollama models selectively without
adding a separate routing-model call:

- Exact Rule/Directive statement, explanation, or rationale questions use a
  deterministic source-grounded path when the exact structured block provides
  the requested answer. No LLM generation is called for that request.
- Direct/simple factual company questions use `llama3.2:3b`.
- Explicit Explain/Describe, Compare/Contrast, Summarize, compound reasoning,
  and clear multi-source synthesis questions use `qwen2.5:7b`.
- Weak/no accepted retrieval evidence still returns exactly
  `Information not found in company knowledge base.` before generation. A
  stronger model is never used to guess around missing evidence.

The routing rules are deterministic Python logic; they do not add another LLM
classification call. Only the model selected for the final grounded generation
is invoked. Rule/Directive deterministic answers continue through the final
output safety gate and preserve source citations.

If the complex model fails at runtime during normal hybrid operation, DocuBot
records a route-fallback event and tries the fast model rather than crashing the
whole request. An explicit `-Model` diagnostic override does not silently switch
models, so controlled model tests remain valid.

Normal run:

```powershell
.\run_docubot_with_log.ps1
```

No special execution flag is required. `DOCUBOT_FAST_LLM_MODEL` and
`DOCUBOT_COMPLEX_LLM_MODEL` are optional advanced overrides; the defaults are
`llama3.2:3b` and `qwen2.5:7b`. The legacy `-Model` parameter remains available
only when one generative model must be forced for diagnostics/A-B testing.

The v6.3 application-code change itself did not require a knowledge-base
rebuild, but the original v6.3 update ZIP accidentally included mutable runtime
index files. v6.3.1 corrects that packaging problem and automatically detects
already-overwritten legacy/generic PDF indexes.

## v6.2.1 Smart Update UI + Short-Fact Retrieval Guardrail

This focused follow-up keeps the v6.2 incremental/cached/batched ingestion
engine intact and closes two validation findings:

- The UI now has one **Update Knowledge Base** action. Normal add/modify/delete
  changes stay incremental. If the embedding/index identity changes or the
  active vector index is missing/unusable, DocuBot detects that automatically
  and shows a confirmation only for that one update before running the existing
  safe staged full rebuild. The permanent Knowledge Base Maintenance panel is
  removed from the normal UI.
- Compact factual source chunks are no longer rejected solely for being under
  25 words when retrieval evidence is exceptionally strong. The exception is
  conservative: the chunk must have high merged retrieval confidence, strong
  direct query-token agreement, and low citation/reference noise. The existing
  reranker threshold and no-evidence fallback are unchanged.

Upgrade from v6.2 requires no `pip install` and no forced knowledge-base
rebuild. Existing vectors remain compatible.

## v6.2 Incremental + Cached + Batched Knowledge Base Indexing

This production-ingestion update is built on the validated v6.1 answer/RAG
baseline. It changes knowledge-base build performance and safety only; retrieval
quality, chunk semantics, current embedding model, reranker, answer behavior,
and LLM A/B support remain unchanged.

### Incremental update behavior

- New files are parsed, chunked, embedded, and added.
- Modified files are reprocessed individually.
- Unchanged files are skipped without parsing, chunking, or embedding.
- Deleted-file chunks are removed from Chroma.
- BM25 is rebuilt from text already stored in Chroma, so unchanged source files
  are not re-read or re-embedded.

### Fast file fingerprinting

The manifest now records file size, nanosecond modification time, SHA256,
parser/index schema, and embedding identity. If size and modification time are
unchanged, the stored SHA256 is reused so normal startup/update checks do not
re-read every large file just to prove it is unchanged. v6.1 manifests are
migrated without forcing a same-model re-index.

### Persistent parsing/chunk cache

Prepared chunks are cached by content hash + index schema. A full rebuild,
recovery, duplicate-content move, or repeated processing can reuse extraction
and chunking work when the source content and chunk schema are unchanged.
Damaged cache files are ignored automatically and the source file is parsed
normally. Cache cleanup is non-critical and never blocks a successful build.

### Batched embeddings and Chroma writes

- The embedding model is loaded once per update/rebuild session.
- Chunks are embedded with the embedding model's batch API when available.
- Chroma vectors are upserted in bulk batches.
- These changes do not alter the generated vector values or retrieval rules.

### Controlled parallel parsing

Independent changed files can be parsed/chunked concurrently. The default is a
conservative maximum of four workers (or fewer on smaller systems), capped at
eight. While one completed document is being batch-embedded, remaining parser
workers can continue processing other files.

Optional override:

```powershell
$env:DOCUBOT_INGEST_WORKERS = "2"
```

Set it to `1` to disable parallel file parsing on a constrained deployment PC.

Optional Chroma write-batch override:

```powershell
$env:DOCUBOT_CHROMA_WRITE_BATCH_SIZE = "128"
```

### Update safety

Changed files are parsed and embedded before the active Chroma index is
mutated. Incremental updates retain old per-file vectors for rollback, BM25 is
written atomically, and the manifest is written atomically. If a commit-stage
failure occurs, DocuBot attempts to restore the previous Chroma/BM25/manifest
state instead of leaving the working knowledge base partially replaced.

Full rebuilds are staged in a temporary Chroma collection and are switched into
the active collection only after parsing, chunking, and embedding have
completed. The previous active collection is retained as a rollback collection
until BM25 and manifest commit successfully.

### Upgrade from v6.1

- No `pip install` is required.
- No full knowledge-base rebuild is required.
- Existing v6.1 vectors remain valid because the embedding model and chunk/index
  schema are unchanged.
- The first startup/update check may enrich the existing manifest with the new
  fast fingerprint fields; this does not re-embed unchanged documents.

## v6.1 Section Explanation + Coverage + QA Matcher Cleanup

This maintenance update keeps the v6 retrieval/index/model behavior intact and
closes three manual-QA gaps before the LLM A/B comparison:

- `Explain Section X` falls back to a grounded section overview when the model
  returns only the section title.
- Grounded explanations of compact label/value documents preserve omitted
  explicit fields such as eligibility, approval, scope, or other short facts.
- The automatic QA matcher tolerates a short inserted middle-name token so an
  expected two-part name such as `Jose Rizal` can match `José Protasio Rizal`
  without making ordinary QA phrases semantically fuzzy.

## v6 Grounded Explanation Depth

Explain/Describe questions are intentionally more complete than direct lookup
questions while remaining restricted to retrieved company knowledge. Direct
Rule/Directive questions stay concise. Exact structured explanations can cover
the requirement, supported amplification/scope, and rationale from the same
exact block.

## LLM Diagnostic / A-B Override

Normal operation is hybrid in v6.3. For a controlled diagnostic run, `-Model`
forces every generative route to one model while deterministic exact structured
answers still skip generation when safe. No knowledge-base rebuild is needed
when switching only the Ollama LLM.

```powershell
.\run_docubot_with_log.ps1 -Model "llama3.2:3b"
```

```powershell
.\run_docubot_with_log.ps1 -Model "qwen2.5:7b"
```

---

## v6.4.25 - Centralized Office LAN Server Foundation

Deployment-only evolution of the Windows-certified v6.4.24 runtime:

- One server PC owns `data/all_documents`, Chroma/BM25 metadata, Ollama, models, embeddings, and reranker.
- Employee PCs use a browser at `http://<server-lan-ip>:8501`; no full DocuBot install is needed on clients.
- `Start_DocuBot_LAN_Server.bat` runs a preflight and binds Streamlit to `0.0.0.0:8501` with XSRF protection retained.
- `Configure_DocuBot_LAN_Firewall.bat` adds a Windows Firewall rule restricted to `Private` profile + `LocalSubnet`.
- LAN employee sessions cannot trigger KB writes. Server-side updates use `Update_DocuBot_Knowledge_Base.bat` while the app is stopped.
- Source/citation buttons become browser downloads in LAN mode, avoiding `os.startfile()` on the server desktop.
- No model, retrieval threshold, reranking, embedding, chunking, answer contract, or hybrid 3B/7B policy change.

See `LAN_DEPLOYMENT_GUIDE.txt`.

## v6.4.46 clean release integrity

- Clean release differs from pristine v6.4.45 by exactly **12 intentional files**: 10 modified + 2 added; zero deleted.
- Release tree contains **1251 files** before ZIP packaging.
- All six Windows QA BAT launchers remain UTF-8 BOM-free.
- Final ZIP is verified by fresh extraction and byte-for-byte comparison before handoff.
---

## v6.4.73 — Phase 7 Performance Baseline

Phase 7 begins with measurement only. The v6.4.72 Chunking-v4 production
baseline remains unchanged. `Run_Phase7_Performance_Baseline.bat` profiles the
known slow paths with the same production models, retrieval thresholds, Top-K,
chunking, prompts, and answer logic.

The benchmark compares first-vs-repeat behavior for the out-of-domain semantic
fallback, OC-007 pointer-arithmetic review, and EN-007 switch review. It records
request/retrieval/LLM latency, lazy vector/reranker load time, Ollama embedding
API time, vector-search time, BM25 time, hybrid-merge time, memory snapshots,
and Ollama residency. Quality guards must remain green during profiling.

The output is written to:

```text
logs\phase7_performance\DocuBot_Phase7_Performance_Baseline_Result_*.zip
```

No optimization should be promoted until that office-side profiling artifact is
reviewed against the certified 16/16 Technical + 19/19 English baseline.

## v6.4.74 — Phase 7 Optimization #1 Trial

This trial removes two measured lower-spec latency wastes while preserving the
v6.4.72/v6.4.73 production-v4 architecture, models, thresholds, Top-K, chunking,
and answer contracts.

- Natural multi-rule MISRA code-review questions may skip Qwen generation only when the
  existing deterministic assessment is itself reference-grounded and passes the
  same semantic-alignment guard used for generated answers. This targets the
  measured EN-007/OC-007 paths where deterministic visible-code states were
  already complete.
- Identity-style OOD questions may stop after BM25 only when the requested
  subject has at least two meaningful tokens and none of those tokens appears
  anywhere in the active company corpus or source titles. Any lexical footprint
  keeps the normal vector/reranker path.
- `Run_Phase7_Optimization1_Trial.bat` replays the exact seven v6.4.73 profiling
  scenarios, retains all quality guards, and additionally requires proof that
  the intended fast paths actually activated.

The trial result is written to:

```text
logs\phase7_optimization1\DocuBot_Phase7_Optimization1_Result_*.zip
```

This is a performance trial, not a final Phase-7 promotion. Full Technical
16/16 + English 19/19 certification remains the quality gate before any
optimized runtime becomes the production lock candidate.
