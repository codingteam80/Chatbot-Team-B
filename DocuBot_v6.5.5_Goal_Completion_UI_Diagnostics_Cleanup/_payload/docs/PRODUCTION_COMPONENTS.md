# Canonical Production Components — v6.5.5 Goal-Completion Release

The following list is authoritative for the active DocuBot production runtime.

| Layer | Component |
|---|---|
| Generation | Qwen2.5:7B (`qwen2.5:7b`) |
| Embedding | Qwen3-Embedding-8B (`qwen3-embedding:8b`) |
| Lexical retrieval | BM25 / `rank_bm25` |
| Vector database | Qdrant v4 |
| Reranking | BAAI/bge-reranker-v2-m3 |
| Chunking | Chunking-v4, 900 size / 150 overlap |
| Active KB | `storage/option_c_qwen3_qdrant_v4` |
| LLM/embedding runtime | Ollama |
| Python runtime | project `venv`, Python 3.11 production line |
| App pipeline | Streamlit → AnswerService → QueryService/Retriever → grounded answer |
| Safety | grounding/semantic guards + strict fallback |
| Performance | Optimization #1 + Optimization #2 |
| Fresh-PC bootstrap | `Setup_DocuBot_Production_Environment.bat` |

## Installation ownership

`requirements.txt` owns Python packages only.

`Setup_DocuBot_Production_Environment.bat` owns the complete deployment bootstrap: Python 3.11 detection/installation, project venv creation, pip requirements, Ollama detection/installation, Qwen model pulls, BGE reranker cache/download, production-setting verification, and read-only KB health.

No setup step automatically rebuilds the production KB.

Superseded vector/embedding/model profiles, rollback-only stores, trial/AB launchers, and promotion harnesses are not active production components. Historical evidence is retained under `evidence/release_history` only.
