# DocuBot — Finalized Production Tree (v6.4.83 Deployment Bootstrap Candidate)

This project tree continues the validated v6.4.82 cleanup/finalization build. v6.4.83 changes only deployment/bootstrap tooling and documentation; it does **not** change retrieval thresholds, Top-K values, Chunking-v4 representation, Optimization #1/#2 behavior, answer logic, or strict fallback.

## Active production components

- **Generation:** `qwen2.5:7b` through Ollama
- **Embeddings:** `qwen3-embedding:8b` through Ollama
- **Lexical retrieval:** BM25 (`rank_bm25`)
- **Vector retrieval:** local Qdrant v4
- **Reranker:** `BAAI/bge-reranker-v2-m3`
- **Chunking:** rule-aware Chunking-v4, 900 / 150
- **Production storage:** `storage/option_c_qwen3_qdrant_v4`
- **Application:** Streamlit + AnswerService/QueryService/ChatManager
- **Safety:** grounding/semantic guards + strict fallback
- **Performance:** certified Optimization #1 and Optimization #2
- **Python runtime:** project `venv` on the certified Python 3.11 line

`llama_index` remains a framework/adapter dependency for the Ollama LLM client. It is **not** a Llama3 model and does not mean Llama3 is part of the production model stack.

## Fresh PC / deployment setup

Run once:

```text
Setup_DocuBot_Production_Environment.bat
```

The setup performs the environment work that `requirements.txt` alone cannot cover:

1. Uses or installs Python 3.11 (WinGet when needed).
2. Creates/reuses the project `venv`.
3. Installs `requirements.txt`.
4. Uses or installs Ollama (WinGet when needed).
5. Pulls `qwen2.5:7b` if missing.
6. Pulls `qwen3-embedding:8b` if missing.
7. Downloads and verifies the `BAAI/bge-reranker-v2-m3` local Hugging Face cache.
8. Verifies Python imports and canonical production settings.
9. Runs the read-only production KB-health check.

The setup does **not** rebuild the production knowledge base.

If WinGet is unavailable and Ollama is missing, install Ollama for Windows manually from the official Ollama download page, then rerun the setup BAT.

## Normal use from VS Code

Open the project folder in VS Code and run:

```text
run.py
```

The repository includes a `.vscode` launch configuration named **DocuBot Production (run.py)**. Normal use does not require certification BAT files.

## Knowledge-base maintenance

`Update_DocuBot_Knowledge_Base.bat` performs the finalized Qdrant-v4 update path. Any source-document change is handled by the certified transactional Qdrant + BM25 rebuild before publication.

`Run_KB_Health_Check.bat` is read-only and checks the current production corpus/index.

## Evidence and history

Historical certification evidence is intentionally preserved. Runtime logs remain under `logs/`; selected architecture/release-history documents are organized under `evidence/`. These files are audit evidence, not active production components.

The actual office/deployment-PC hardware acceptance remains deferred until that machine is available. Manual real-user QA remains the next functional gate before Golden Release.
