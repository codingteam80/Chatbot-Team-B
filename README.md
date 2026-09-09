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

## LLM A/B Test Switching

The default model remains `llama3.2:3b`. No knowledge-base rebuild is needed
when switching only the Ollama LLM.

```powershell
.\run_docubot_with_log.ps1 -Model "llama3.2:3b"
```

```powershell
.\run_docubot_with_log.ps1 -Model "qwen2.5:7b"
```
