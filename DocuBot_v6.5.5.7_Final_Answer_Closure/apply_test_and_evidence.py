from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAYLOAD = HERE / "payload"
LOG_ROOT = ROOT / "logs" / "v6_5_5_7_closure"

TARGETS = [
    (PAYLOAD / "services" / "answer_service.py", ROOT / "services" / "answer_service.py"),
    (PAYLOAD / "services" / "verified_answer_contract.py", ROOT / "services" / "verified_answer_contract.py"),
    (PAYLOAD / "scripts" / "test_verified_answer_contract_v6_5_5_7.py", ROOT / "scripts" / "test_verified_answer_contract_v6_5_5_7.py"),
    (PAYLOAD / "scripts" / "test_llm_verified_answer_contract_v6_5_5_7.py", ROOT / "scripts" / "test_llm_verified_answer_contract_v6_5_5_7.py"),
    (PAYLOAD / "scripts" / "generate_chunking_evidence_v6_5_5_7.py", ROOT / "scripts" / "generate_chunking_evidence_v6_5_5_7.py"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_stage(name: str, command: list[str]) -> dict:
    print("\n" + "=" * 76)
    print(name)
    print("=" * 76)
    started = datetime.now()
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace")
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip())
    return {
        "name": name,
        "command": command,
        "exit_code": proc.returncode,
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def newest(pattern: str) -> Path | None:
    items = list(ROOT.glob(pattern))
    return max(items, key=lambda p: p.stat().st_mtime) if items else None


def main() -> int:
    if not (ROOT / "services" / "answer_service.py").exists():
        print(f"[FAIL] This folder must be directly inside the live DocuBot project: {ROOT}")
        return 2
    if not (ROOT / "storage" / "option_c_qwen3_qdrant_v4" / "bm25" / "corpus.pkl").exists():
        print("[FAIL] Active production BM25 corpus not found. Wrong project root or incomplete project.")
        return 2

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = LOG_ROOT / f"backup_before_v6.5.5.7_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    previous = {}
    print(f"[PASS] Project root : {ROOT}")
    print(f"[PASS] Python       : {sys.executable}")
    print("[INFO] Applying only the final answer-contract closure + read-only evidence tools.")
    print("[INFO] Retrieval, models, chunk size/overlap, Top-K, thresholds, and KB updater are not changed.")

    for src, dst in TARGETS:
        if not src.exists():
            print(f"[FAIL] Missing payload file: {src}")
            return 2
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            backup = backup_dir / dst.relative_to(ROOT)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, backup)
            previous[str(dst)] = str(backup)
        else:
            previous[str(dst)] = None
        shutil.copy2(src, dst)
        print(f"[APPLY] {dst.relative_to(ROOT)}")

    stages = []
    stages.append(run_stage(
        "Python compile / import safety",
        [sys.executable, "-m", "py_compile",
         "services/answer_service.py",
         "services/verified_answer_contract.py",
         "scripts/test_verified_answer_contract_v6_5_5_7.py",
         "scripts/test_llm_verified_answer_contract_v6_5_5_7.py",
         "scripts/generate_chunking_evidence_v6_5_5_7.py"],
    ))

    validator = ROOT / "scripts" / "validate_v6_5_5_goal_completion.py"
    validator_manifest = ROOT / "FINAL_COMPONENT_MANIFEST_v6.5.5.json"
    if validator.exists() and validator_manifest.exists():
        stages.append(run_stage("Existing v6.5.5 architecture validator", [sys.executable, str(validator)]))
    else:
        stages.append({
            "name": "Existing v6.5.5 architecture validator",
            "exit_code": 0,
            "status": "SKIPPED",
            "reason": "validator or its v6.5.5 manifest is not present in this live tree",
        })
        print("\n[INFO] Existing 77-check validator skipped because its final manifest is not present.")

    stages.append(run_stage(
        "Deterministic verified-answer contract unit checks",
        [sys.executable, "scripts/test_verified_answer_contract_v6_5_5_7.py"],
    ))
    stages.append(run_stage(
        "Read-only active chunking evidence",
        [sys.executable, "scripts/generate_chunking_evidence_v6_5_5_7.py"],
    ))
    stages.append(run_stage(
        "Live LLM answer-closure retest over previously verified retrieval contexts",
        [sys.executable, "scripts/test_llm_verified_answer_contract_v6_5_5_7.py"],
    ))

    failed = [stage for stage in stages if int(stage.get("exit_code", 0) or 0) != 0]
    rollback = bool(failed)
    if rollback:
        print("\n[FAIL] One or more closure stages failed. Restoring the prior production files.")
        for src, dst in reversed(TARGETS):
            backup_value = previous.get(str(dst))
            if backup_value:
                shutil.copy2(Path(backup_value), dst)
            else:
                try:
                    dst.unlink()
                except FileNotFoundError:
                    pass
        print("[PASS] Prior production state restored.")
    else:
        print("\n[PASS] All closure stages passed. v6.5.5.7 answer contract remains applied.")

    chunk_zip = newest("logs/chunking_evidence/DocuBot_v6.5.5.7_Chunking_Evidence_*.zip")
    chunk_sha = Path(str(chunk_zip) + ".sha256.txt") if chunk_zip else None
    answer_json = newest("logs/v6_5_5_7_closure/answer_contract_live_*.json")
    validation_json = ROOT / "logs" / "v6_5_5_completion" / "validation_latest.json"

    summary = {
        "version": "v6.5.5.7",
        "created": datetime.now().isoformat(timespec="seconds"),
        "overall": "FAIL" if failed else "PASS",
        "patch_kept": not rollback,
        "rollback_performed": rollback,
        "failed_stages": [stage.get("name") for stage in failed],
        "stages": stages,
        "answer_contract_live_json": str(answer_json) if answer_json else "",
        "chunking_evidence_zip": str(chunk_zip) if chunk_zip else "",
        "chunking_evidence_sha256": str(chunk_sha) if chunk_sha and chunk_sha.exists() else "",
        "unchanged_architecture": {
            "retrieval_logic_changed": False,
            "embedding_model_changed": False,
            "generation_model_changed": False,
            "chunking_parameters_changed": False,
            "top_k_changed": False,
            "threshold_changed": False,
            "kb_updater_changed": False,
        },
    }
    summary_path = LOG_ROOT / f"final_answer_closure_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    result_zip = LOG_ROOT / f"DocuBot_v6.5.5.7_Final_Answer_Closure_Result_{stamp}.zip"
    files = [summary_path]
    if answer_json and answer_json.exists():
        files.append(answer_json)
    if validation_json.exists():
        files.append(validation_json)
    if chunk_zip and chunk_zip.exists():
        files.append(chunk_zip)
    if chunk_sha and chunk_sha.exists():
        files.append(chunk_sha)
    with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            zf.write(path, arcname=path.name)
    result_sha = Path(str(result_zip) + ".sha256.txt")
    result_sha.write_text(f"{sha256(result_zip)}  {result_zip.name}\n", encoding="utf-8")

    print("\n" + "=" * 76)
    print("DOCUBOT v6.5.5.7 FINAL ANSWER CLOSURE")
    print("=" * 76)
    print(f"Overall        : {summary['overall']}")
    print(f"Patch kept     : {summary['patch_kept']}")
    print(f"Result ZIP     : {result_zip}")
    print(f"Result SHA256  : {result_sha}")
    if chunk_zip:
        print(f"Chunk evidence : {chunk_zip}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
