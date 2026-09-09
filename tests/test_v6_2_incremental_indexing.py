import sys
import types
from pathlib import Path
from unittest.mock import patch


def _cache_resource(*args, **kwargs):
    def decorator(function):
        function.clear = lambda: None
        return function
    return decorator


streamlit_stub = sys.modules.get("streamlit") or types.ModuleType("streamlit")
streamlit_stub.cache_resource = _cache_resource
sys.modules.setdefault("streamlit", streamlit_stub)

chromadb_stub = sys.modules.get("chromadb") or types.ModuleType("chromadb")
chromadb_stub.PersistentClient = object
sys.modules.setdefault("chromadb", chromadb_stub)

rank_bm25_stub = sys.modules.get("rank_bm25") or types.ModuleType("rank_bm25")


class _FakeBM25:
    def __init__(self, corpus):
        self.corpus = corpus


rank_bm25_stub.BM25Okapi = _FakeBM25
sys.modules.setdefault("rank_bm25", rank_bm25_stub)

llama_index_stub = sys.modules.get("llama_index") or types.ModuleType("llama_index")
llama_embeddings_stub = sys.modules.get("llama_index.embeddings") or types.ModuleType("llama_index.embeddings")
llama_hf_stub = sys.modules.get("llama_index.embeddings.huggingface") or types.ModuleType("llama_index.embeddings.huggingface")
llama_hf_stub.HuggingFaceEmbedding = object
sys.modules.setdefault("llama_index", llama_index_stub)
sys.modules.setdefault("llama_index.embeddings", llama_embeddings_stub)
sys.modules.setdefault("llama_index.embeddings.huggingface", llama_hf_stub)
llama_core_stub = sys.modules.get("llama_index.core") or types.ModuleType("llama_index.core")
llama_node_parser_stub = sys.modules.get("llama_index.core.node_parser") or types.ModuleType("llama_index.core.node_parser")


class _SentenceSplitter:
    def __init__(self, chunk_size=900, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        return [text] if text else []


llama_node_parser_stub.SentenceSplitter = _SentenceSplitter
sys.modules.setdefault("llama_index.core", llama_core_stub)
sys.modules.setdefault("llama_index.core.node_parser", llama_node_parser_stub)

unstructured_stub = sys.modules.get("unstructured") or types.ModuleType("unstructured")
unstructured_partition_stub = sys.modules.get("unstructured.partition") or types.ModuleType("unstructured.partition")
unstructured_auto_stub = sys.modules.get("unstructured.partition.auto") or types.ModuleType("unstructured.partition.auto")
unstructured_auto_stub.partition = lambda *args, **kwargs: []
sys.modules.setdefault("unstructured", unstructured_stub)
sys.modules.setdefault("unstructured.partition", unstructured_partition_stub)
sys.modules.setdefault("unstructured.partition.auto", unstructured_auto_stub)

from config.settings import DEFAULT_BATCH_SIZE
from ingestion.ingest import IngestionPipeline
from scripts.build_index import prepare_records_for_storage
from scripts.smart_build import compare_manifests
from utils.hash_utils import FileHasher
from utils.ingestion_cache import IngestionCache
from utils.manifest import ManifestManager


def _record(index):
    return {
        "text": f"chunk {index}",
        "metadata": {
            "file_path": "C:/docs/policy.txt",
            "file_name": "policy.txt",
            "chunk_id": index,
        },
    }


def test_manifest_reuses_sha_when_size_and_mtime_are_unchanged(tmp_path):
    source = tmp_path / "policy.txt"
    source.write_text("company policy content", encoding="utf-8")
    stat = FileHasher.stat_fingerprint(source)
    signature = ManifestManager.current_index_signature()
    key = ManifestManager.document_key(source)
    previous = {
        key: {
            "file_name": source.name,
            "file_path": str(source.resolve()),
            "indexed_file_path": str(source.resolve()),
            "hash": "already-computed-hash",
            **stat,
            **signature,
            "last_indexed": "2026-09-09 10:00:00",
        }
    }

    with patch.object(FileHasher, "sha256", side_effect=AssertionError("SHA should be reused")):
        manifest = ManifestManager.build([source], previous_manifest=previous)

    assert manifest[key]["hash"] == "already-computed-hash"
    assert manifest[key]["last_indexed"] == "2026-09-09 10:00:00"


def test_manifest_embedding_identity_change_marks_document_updated(tmp_path):
    source = tmp_path / "policy.txt"
    source.write_text("company policy content", encoding="utf-8")
    stat = FileHasher.stat_fingerprint(source)
    file_hash = FileHasher.sha256(source)
    key = ManifestManager.document_key(source)
    current = ManifestManager.current_index_signature()
    old = {
        key: {
            "file_path": str(source.resolve()),
            "indexed_file_path": str(source.resolve()),
            "file_name": source.name,
            "hash": file_hash,
            **stat,
            "index_schema_version": current["index_schema_version"],
            "embedding_model": "different-embedding-model",
            "embedding_normalize": current["embedding_normalize"],
            "last_indexed": "2026-09-09 10:00:00",
        }
    }

    new = ManifestManager.build([source], previous_manifest=old)
    changes = compare_manifests(old, new)

    assert changes["updated"] == [key]
    assert changes["unchanged"] == []


def test_ingestion_cache_round_trip_and_prune(tmp_path, monkeypatch):
    import utils.ingestion_cache as cache_module

    monkeypatch.setattr(cache_module, "INGESTION_CACHE_DIR", tmp_path / "cache")
    file_hash = "a" * 64
    prepared = [
        {
            "text": "Vacation Leave\nEmployees receive leave.",
            "metadata": {"parser_strategy": "generic_sentence"},
        }
    ]

    IngestionCache.save_ready(file_hash, prepared)
    loaded = IngestionCache.load(file_hash)

    assert loaded["status"] == "ready"
    assert loaded["prepared_chunks"] == prepared
    assert IngestionCache.prune({file_hash}) == 0
    assert IngestionCache.prune(set()) == 1


def test_cached_prepared_chunks_bypass_document_parser(tmp_path, monkeypatch):
    source = tmp_path / "policy.txt"
    source.write_text("placeholder", encoding="utf-8")
    cached = {
        "status": "ready",
        "prepared_chunks": [
            {
                "text": "Employee Leave Policy\nAll leave requests require manager approval.",
                "metadata": {"parser_strategy": "generic_sentence"},
            }
        ],
    }

    monkeypatch.setattr(IngestionCache, "load", lambda file_hash: cached)
    pipeline = IngestionPipeline(use_cache=True)
    monkeypatch.setattr(
        pipeline.parser,
        "parse",
        lambda *_: (_ for _ in ()).throw(AssertionError("parser should not run")),
    )

    records = pipeline.process_document(source, file_hash="b" * 64)

    assert pipeline.last_cache_hit is True
    assert len(records) == 1
    assert records[0]["metadata"]["file_name"] == "policy.txt"


class _BatchEmbeddingModel:
    def __init__(self):
        self.calls = []

    def get_text_embedding_batch(self, texts):
        self.calls.append(list(texts))
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


def test_embeddings_are_generated_in_batches_not_per_chunk():
    model = _BatchEmbeddingModel()
    records = [_record(index) for index in range(DEFAULT_BATCH_SIZE * 2 + 5)]

    payload, failed = prepare_records_for_storage(records, model)

    assert failed == 0
    assert len(payload["embeddings"]) == len(records)
    assert len(model.calls) == 3
    assert len(model.calls[0]) == DEFAULT_BATCH_SIZE
    assert len(model.calls[1]) == DEFAULT_BATCH_SIZE
    assert len(model.calls[2]) == 5


def test_old_v6_1_manifest_without_embedding_fields_does_not_force_reindex(tmp_path):
    source = tmp_path / "policy.txt"
    source.write_text("company policy content", encoding="utf-8")
    file_hash = FileHasher.sha256(source)
    key = ManifestManager.document_key(source)
    old = {
        key: {
            "file_path": str(source.resolve()),
            "indexed_file_path": str(source.resolve()),
            "file_name": source.name,
            "hash": file_hash,
            "index_schema_version": ManifestManager.current_index_signature()["index_schema_version"],
            "last_indexed": "2026-09-09 10:00:00",
        }
    }

    new = ManifestManager.build([source], previous_manifest=old)
    changes = compare_manifests(old, new)

    assert changes["updated"] == []
    assert changes["unchanged"] == [key]


def test_incremental_no_change_path_does_not_parse_or_embed(monkeypatch, tmp_path):
    import scripts.smart_build as smart_module

    source = tmp_path / "policy.txt"
    source.write_text("company policy content", encoding="utf-8")
    key = ManifestManager.document_key(source)
    signature = ManifestManager.current_index_signature()
    stat = FileHasher.stat_fingerprint(source)
    manifest = {
        key: {
            "file_name": source.name,
            "file_path": str(source.resolve()),
            "indexed_file_path": str(source.resolve()),
            "hash": FileHasher.sha256(source),
            **stat,
            **signature,
            "last_indexed": "2026-09-09 10:00:00",
        }
    }

    class _Collection:
        def count(self):
            return 1

    monkeypatch.setattr(smart_module, "get_all_documents", lambda: [source])
    monkeypatch.setattr(smart_module.ManifestManager, "load", lambda: manifest)
    monkeypatch.setattr(
        smart_module.ManifestManager,
        "build",
        lambda documents, previous_manifest=None: manifest,
    )
    monkeypatch.setattr(smart_module, "get_collection", lambda: _Collection())
    monkeypatch.setattr(smart_module, "bm25_is_ready", lambda: True)
    monkeypatch.setattr(smart_module.IngestionCache, "prune", lambda hashes: 0)
    monkeypatch.setattr(
        smart_module,
        "_prepare_changed_documents",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unchanged files must not be parsed or embedded")
        ),
    )

    assert smart_module.smart_build() is True


def test_changed_file_preparation_failure_keeps_active_index_unmodified(monkeypatch, tmp_path):
    import scripts.smart_build as smart_module

    source = tmp_path / "policy.txt"
    source.write_text("new policy content", encoding="utf-8")
    key = ManifestManager.document_key(source)
    signature = ManifestManager.current_index_signature()
    old = {
        key: {
            "file_name": source.name,
            "file_path": str(source.resolve()),
            "indexed_file_path": str(source.resolve()),
            "hash": "old-hash",
            **signature,
            "last_indexed": "2026-09-09 10:00:00",
        }
    }
    new = {
        key: {
            "file_name": source.name,
            "file_path": str(source.resolve()),
            "indexed_file_path": str(source.resolve()),
            "hash": "new-hash",
            **signature,
            "last_indexed": "2026-09-09 11:00:00",
        }
    }

    class _Collection:
        mutations = 0

        def count(self):
            return 1

        def delete(self, **kwargs):
            self.mutations += 1

        def upsert(self, **kwargs):
            self.mutations += 1

    collection = _Collection()
    monkeypatch.setattr(smart_module, "get_all_documents", lambda: [source])
    monkeypatch.setattr(smart_module.ManifestManager, "load", lambda: old)
    monkeypatch.setattr(
        smart_module.ManifestManager,
        "build",
        lambda documents, previous_manifest=None: new,
    )
    monkeypatch.setattr(smart_module, "get_collection", lambda: collection)
    monkeypatch.setattr(
        smart_module,
        "_prepare_changed_documents",
        lambda changes, new_manifest: ({}, [(source.name, "parse failed")], 0),
    )

    assert smart_module.smart_build() is False
    assert collection.mutations == 0


def test_full_rebuild_collection_swap_can_be_rolled_back():
    from scripts.build_index import _rollback_collection_swap, _swap_staging_collection

    class _Collection:
        def __init__(self, client, name):
            self.client = client
            self.name = name

        def modify(self, name):
            del self.client.collections[self.name]
            self.name = name
            self.client.collections[name] = self

    class _Client:
        def __init__(self):
            self.collections = {}

        def add(self, name):
            collection = _Collection(self, name)
            self.collections[name] = collection
            return collection

        def get_collection(self, name):
            if name not in self.collections:
                raise KeyError(name)
            return self.collections[name]

        def delete_collection(self, name):
            if name not in self.collections:
                raise KeyError(name)
            del self.collections[name]

    client = _Client()
    original = client.add("company_knowledge")
    client.add("staging")

    had_active = _swap_staging_collection(
        client,
        "staging",
        "company_knowledge",
        "backup",
    )

    assert had_active is True
    assert client.collections["backup"] is original
    assert client.collections["company_knowledge"] is not original

    _rollback_collection_swap(
        client,
        "company_knowledge",
        "backup",
        "failed",
        had_active,
    )

    assert client.collections["company_knowledge"] is original
    assert "backup" not in client.collections
    assert "failed" not in client.collections


def test_interrupted_full_rebuild_backup_is_recovered_when_active_missing():
    from scripts.build_index import recover_active_collection_if_needed

    class _Collection:
        def __init__(self, client, name):
            self.client = client
            self.name = name

        def modify(self, name):
            del self.client.collections[self.name]
            self.name = name
            self.client.collections[name] = self

    class _Client:
        def __init__(self):
            self.collections = {}

        def add(self, name):
            item = _Collection(self, name)
            self.collections[name] = item
            return item

        def get_collection(self, name):
            if name not in self.collections:
                raise KeyError(name)
            return self.collections[name]

        def list_collections(self):
            return list(self.collections.values())

    client = _Client()
    backup = client.add("company_knowledge__backup__20260909160500_abcd")

    recovered = recover_active_collection_if_needed(client)

    assert recovered is backup
    assert client.collections["company_knowledge"] is backup


def test_bm25_commit_failure_rolls_back_new_incremental_vectors(monkeypatch, tmp_path):
    import scripts.smart_build as smart_module

    source = tmp_path / "new_policy.txt"
    source.write_text("new company policy content long enough for testing", encoding="utf-8")
    key = ManifestManager.document_key(source)
    signature = ManifestManager.current_index_signature()
    new_manifest = {
        key: {
            "file_name": source.name,
            "file_path": str(source.resolve()),
            "indexed_file_path": str(source.resolve()),
            "hash": "new-hash",
            **signature,
            "last_indexed": "2026-09-09 11:00:00",
        }
    }
    payload = {
        "ids": ["new-id"],
        "documents": ["new company policy content"],
        "metadatas": [{"file_path": str(source.resolve()), "chunk_id": 0}],
        "embeddings": [[0.1, 0.2]],
        "records": [{"text": "new company policy content", "metadata": {"file_path": str(source.resolve()), "chunk_id": 0}}],
    }

    class _Collection:
        def __init__(self):
            self.ids = set()

        def count(self):
            return len(self.ids)

        def upsert(self, ids, **kwargs):
            self.ids.update(ids)

        def delete(self, ids):
            self.ids.difference_update(ids)

    collection = _Collection()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    monkeypatch.setattr(smart_module, "get_all_documents", lambda: [source])
    monkeypatch.setattr(smart_module.ManifestManager, "load", lambda: {})
    monkeypatch.setattr(
        smart_module.ManifestManager,
        "build",
        lambda documents, previous_manifest=None: new_manifest,
    )
    monkeypatch.setattr(smart_module, "get_collection", lambda: collection)
    monkeypatch.setattr(
        smart_module,
        "_prepare_changed_documents",
        lambda changes, manifest: (
            {
                key: {
                    "change_type": "added",
                    "file_name": source.name,
                    "status": "ready",
                    "skip_reason": None,
                    "payload": payload,
                    "record_count": 1,
                }
            },
            [],
            0,
        ),
    )
    monkeypatch.setattr(smart_module, "snapshot_bm25_resources", lambda: snapshot)
    monkeypatch.setattr(smart_module, "discard_bm25_snapshot", lambda path: None)
    monkeypatch.setattr(smart_module, "restore_bm25_resources", lambda path: None)
    monkeypatch.setattr(
        smart_module,
        "rebuild_bm25_from_chroma",
        lambda collection: (_ for _ in ()).throw(RuntimeError("BM25 failed")),
    )
    monkeypatch.setattr(smart_module.ManifestManager, "save", lambda manifest: None)

    assert smart_module.smart_build() is False
    assert collection.ids == set()


def test_update_plan_requires_full_rebuild_for_global_embedding_identity_change(monkeypatch, tmp_path):
    import scripts.smart_build as smart_module

    source = tmp_path / "policy.txt"
    source.write_text("company policy content", encoding="utf-8")
    key = ManifestManager.document_key(source)
    stat = FileHasher.stat_fingerprint(source)
    current = ManifestManager.current_index_signature()
    old = {
        key: {
            "file_name": source.name,
            "file_path": str(source.resolve()),
            "indexed_file_path": str(source.resolve()),
            "hash": FileHasher.sha256(source),
            **stat,
            "index_schema_version": current["index_schema_version"],
            "embedding_model": "legacy-embedding-model",
            "embedding_normalize": current["embedding_normalize"],
            "last_indexed": "2026-09-09 10:00:00",
        }
    }

    class _Collection:
        def count(self):
            return 1

    monkeypatch.setattr(smart_module, "get_all_documents", lambda: [source])
    monkeypatch.setattr(smart_module.ManifestManager, "load", lambda: old)
    monkeypatch.setattr(smart_module, "get_collection", lambda: _Collection())
    monkeypatch.setattr(smart_module, "bm25_is_ready", lambda: True)

    plan = smart_module.get_update_plan()

    assert plan["mode"] == "full_rebuild"
    assert "embedding/index identity changed" in plan["reason"]


def test_update_plan_keeps_normal_file_change_incremental(monkeypatch, tmp_path):
    import scripts.smart_build as smart_module

    source = tmp_path / "policy.txt"
    source.write_text("new company policy content", encoding="utf-8")
    key = ManifestManager.document_key(source)
    signature = ManifestManager.current_index_signature()
    stat = FileHasher.stat_fingerprint(source)
    old = {
        key: {
            "file_name": source.name,
            "file_path": str(source.resolve()),
            "indexed_file_path": str(source.resolve()),
            "hash": "old-hash",
            **stat,
            "modified_time_ns": stat["modified_time_ns"] - 1,
            **signature,
            "last_indexed": "2026-09-09 10:00:00",
        }
    }

    class _Collection:
        def count(self):
            return 1

    monkeypatch.setattr(smart_module, "get_all_documents", lambda: [source])
    monkeypatch.setattr(smart_module.ManifestManager, "load", lambda: old)
    monkeypatch.setattr(smart_module, "get_collection", lambda: _Collection())
    monkeypatch.setattr(smart_module, "bm25_is_ready", lambda: True)

    plan = smart_module.get_update_plan()

    assert plan["mode"] == "incremental"
    assert plan["reason"] is None
    assert plan["changes"]["updated"] == [key]


def test_main_ui_has_one_smart_update_button_and_no_permanent_maintenance_panel():
    project_root = Path(__file__).resolve().parents[1]
    app_text = (project_root / "app.py").read_text(encoding="utf-8")

    assert "Knowledge Base Maintenance" not in app_text
    assert "Full Rebuild Knowledge Base" not in app_text
    assert '"Update Knowledge Base"' in app_text
    assert "kb_smart_full_rebuild_confirm" in app_text
