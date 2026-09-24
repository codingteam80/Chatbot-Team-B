from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAYLOAD = HERE / "payload"
LOG_DIR = ROOT / "logs" / "v6_5_5_8_calibration"
LOG_DIR.mkdir(parents=True, exist_ok=True)
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_DIR = LOG_DIR / f"backup_{STAMP}"
RESULT_JSON = LOG_DIR / f"natural_query_calibration_{STAMP}.json"
RESULT_ZIP = LOG_DIR / f"DocuBot_v6.5.5.8_Natural_Query_Calibration_Result_{STAMP}.zip"
RESULT_SHA = RESULT_ZIP.with_suffix(RESULT_ZIP.suffix + ".sha256.txt")

FILES = [
    Path("services/answer_service.py"),
    Path("services/misra_compliance.py"),
    Path("retrieval/retriever.py"),
    Path("ingestion/pdf_structure.py"),
    Path("scripts/validate_v6_5_5_8_natural_query_calibration.py"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_stage(name: str, cmd: list[str]) -> dict:
    started = datetime.now().isoformat(timespec="seconds")
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    finished = datetime.now().isoformat(timespec="seconds")
    stage = {
        "name": name,
        "command": cmd,
        "exit_code": proc.returncode,
        "started": started,
        "finished": finished,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    print(f"\n=== {name} ===")
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip())
    print(f"Exit code: {proc.returncode}")
    return stage


def choose_python() -> str:
    venv_python = ROOT / "venv" / "Scripts" / "python.exe"
    return str(venv_python if venv_python.is_file() else Path(sys.executable))


def backup_current() -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for rel in FILES:
        src = ROOT / rel
        if src.is_file():
            hashes[str(rel)] = sha256(src)
            dst = BACKUP_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            hashes[str(rel)] = None
    return hashes


def apply_payload() -> dict[str, str]:
    after = {}
    for rel in FILES:
        src = PAYLOAD / rel
        if not src.is_file():
            raise FileNotFoundError(f"Missing payload file: {rel}")
        dst = ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        after[str(rel)] = sha256(dst)
    return after


def rollback() -> None:
    for rel in FILES:
        backup = BACKUP_DIR / rel
        target = ROOT / rel
        if backup.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, target)
        else:
            try:
                target.unlink()
            except FileNotFoundError:
                pass


def package(result: dict) -> None:
    RESULT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(RESULT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(RESULT_JSON, arcname=RESULT_JSON.name)
        for extra in [HERE / "README_v6.5.5.8.txt", HERE / "CALIBRATION_SET_A_CRITICAL_RETEST.txt"]:
            if extra.is_file():
                zf.write(extra, arcname=extra.name)
        validator = ROOT / "scripts" / "validate_v6_5_5_8_natural_query_calibration.py"
        if validator.is_file():
            zf.write(validator, arcname="evidence/validate_v6_5_5_8_natural_query_calibration.py")
    digest = sha256(RESULT_ZIP)
    RESULT_SHA.write_text(f"{digest}  {RESULT_ZIP.name}\n", encoding="utf-8")


def main() -> int:
    py = choose_python()
    result = {
        "version": "v6.5.5.8",
        "created": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "overall": "FAIL",
        "patch_kept": False,
        "rollback_performed": False,
        "production_parameters_changed": {
            "embedding_model": False,
            "generation_model": False,
            "reranker_model": False,
            "chunk_size_overlap": False,
            "global_threshold": False,
            "top_k": False,
            "qdrant_vectors": False,
            "kb_updater_transaction_logic": False,
        },
        "before_sha256": {},
        "after_sha256": {},
        "stages": [],
        "failed_stages": [],
    }

    try:
        print("DocuBot v6.5.5.8 Natural Query Calibration")
        print("Project root:", ROOT)
        print("Python:", py)
        result["before_sha256"] = backup_current()
        result["after_sha256"] = apply_payload()

        compile_files = [str(rel) for rel in FILES if rel.suffix == ".py"]
        result["stages"].append(run_stage(
            "Python compile/import safety",
            [py, "-m", "py_compile", *compile_files],
        ))

        goal_validator = ROOT / "scripts" / "validate_v6_5_5_goal_completion.py"
        if goal_validator.is_file():
            result["stages"].append(run_stage(
                "Existing v6.5.5 architecture validator",
                [py, str(goal_validator.relative_to(ROOT))],
            ))

        answer_contract = ROOT / "scripts" / "test_verified_answer_contract_v6_5_5_7.py"
        if answer_contract.is_file():
            result["stages"].append(run_stage(
                "Existing v6.5.5.7 verified-answer contract",
                [py, str(answer_contract.relative_to(ROOT))],
            ))

        result["stages"].append(run_stage(
            "v6.5.5.8 deterministic natural-query calibration validator",
            [py, "scripts/validate_v6_5_5_8_natural_query_calibration.py"],
        ))

        failed = [s["name"] for s in result["stages"] if int(s.get("exit_code", 1)) != 0]
        result["failed_stages"] = failed
        if failed:
            rollback()
            result["rollback_performed"] = True
            result["patch_kept"] = False
            result["overall"] = "FAIL"
        else:
            result["patch_kept"] = True
            result["overall"] = "PASS"

    except Exception as error:
        result["fatal_error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        try:
            rollback()
            result["rollback_performed"] = True
        except Exception as rollback_error:
            result["rollback_error"] = f"{type(rollback_error).__name__}: {rollback_error}"
    finally:
        package(result)

    print("\n" + "=" * 72)
    print("Overall        :", result["overall"])
    print("Patch kept     :", result["patch_kept"])
    print("Rollback       :", result["rollback_performed"])
    print("Result ZIP     :", RESULT_ZIP)
    print("SHA256         :", RESULT_SHA)
    print("=" * 72)
    return 0 if result["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
