@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title DocuBot - KB Update Portability Dry Run
chcp 65001 >nul

set "PYTHON_EXE="
if exist "venv\Scripts\python.exe" set "PYTHON_EXE=venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "env\Scripts\python.exe" set "PYTHON_EXE=env\Scripts\python.exe"

if not defined PYTHON_EXE (
  echo [FAIL] Project Python environment not found.
  echo [INFO] Run Setup_DocuBot_Production_Environment.bat first.
  pause
  exit /b 2
)

echo ============================================================
echo DocuBot - KB Update Portability Dry Run
echo ============================================================
echo [INFO] This test does NOT rebuild Qdrant/BM25.
echo [INFO] It checks Python/dependencies, Qdrant access, Ollama embedding readiness,
echo [INFO] filesystem permissions, disk space, project cwd, and update locking.
echo.

"%PYTHON_EXE%" -X utf8 -u -m scripts.test_kb_update_portability
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [PASS] KB Update portability dry run passed.
) else (
  echo [FAIL] KB Update portability dry run failed.
  echo [INFO] Review logs\kb_update_portability\kb_update_portability_latest.json
)
echo.
pause
exit /b %RC%
