@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title DocuBot v6.4.82 - Final Cleanup Validation
chcp 65001 >nul

set "PYTHON_EXE="
if exist "venv\Scripts\python.exe" set "PYTHON_EXE=venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if not defined PYTHON_EXE (
  echo [FAIL] Project Python environment not found.
  echo Expected venv\Scripts\python.exe or .venv\Scripts\python.exe.
  pause
  exit /b 2
)

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ======================================================================
echo DocuBot v6.4.82 - CLEANUP / FINALIZATION LOCAL VALIDATION
echo ======================================================================
echo This validates the cleaned production tree and preserves historical evidence.
echo It does not call Qwen generation and does not rebuild the KB.
echo.

"%PYTHON_EXE%" -X utf8 -u -m scripts.validate_v6_4_82_cleanup
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [PASS] Cleanup/finalization structure validation passed.
  echo Next: start run.py from VS Code and do the manual question QA checklist.
) else (
  echo [FAIL] Cleanup/finalization validation found an issue.
)
echo Report: logs\cleanup_finalization\v6.4.82_cleanup_validation_latest.json
pause
exit /b %RC%
