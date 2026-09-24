from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config.settings import BM25_DIR, EMBED_MODEL_NAME
from retrieval.semantic_rule_resolver import (
    SEMANTIC_RULE_INDEX_SCHEMA,
    SEMANTIC_RULE_METADATA,
    SEMANTIC_RULE_VECTORS,
    _load_bm25_records,
    build_semantic_rule_index,
    extract_rule_profiles,
    semantic_rule_index_ready,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build DocuBot MISRA Rule-level semantic index")
    parser.add_argument("--check", action="store_true", help="Validate source/profile extraction only; do not call Ollama or write the index.")
    parser.add_argument("--force", action="store_true", help="Rebuild even when the current derived index is valid.")
    args = parser.parse_args()

    records = _load_bm25_records()
    profiles = extract_rule_profiles(records)
    summary = {
        "schema": SEMANTIC_RULE_INDEX_SCHEMA,
        "bm25_corpus": str(BM25_DIR / "corpus.pkl"),
        "record_count": len(records),
        "profile_count": len(profiles),
        "embedding_model": EMBED_MODEL_NAME,
        "metadata_path": str(SEMANTIC_RULE_METADATA),
        "vectors_path": str(SEMANTIC_RULE_VECTORS),
        "index_ready_before": semantic_rule_index_ready(),
    }

    if len(profiles) < 10:
        print(json.dumps({**summary, "ok": False, "error": "insufficient structured profiles"}, indent=2))
        return 2

    if args.check:
        required_refs = {"Rule 2.7", "Rule 13.5", "Rule 15.3", "Rule 16.4", "Rule 21.19"}
        present = {profile.display_name for profile in profiles}
        missing = sorted(required_refs - present)
        ok = not missing
        print(json.dumps({**summary, "ok": ok, "check_only": True, "missing_reference_samples": missing}, indent=2))
        return 0 if ok else 3

    try:
        result = build_semantic_rule_index(force=args.force)
    except Exception as error:
        print(json.dumps({**summary, "ok": False, "error": f"{type(error).__name__}: {error}"}, indent=2))
        return 4

    print(json.dumps({**summary, "ok": True, "build": result, "index_ready_after": semantic_rule_index_ready()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
