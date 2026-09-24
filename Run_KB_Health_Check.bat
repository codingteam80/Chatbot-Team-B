@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo DocuBot v6.4.82 - PRODUCTION KNOWLEDGE BASE HEALTH CHECK
echo Read-only: no rebuild, no embeddings, no Ollama generation
echo ============================================================
echo.

set "PYTHON_EXE="
if exist "venv\Scripts\python.exe" set "PYTHON_EXE=venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if not defined PYTHON_EXE (
  echo [FAIL] Project Python environment not found.
  echo Expected venv\Scripts\python.exe or .venv\Scripts\python.exe.
  pause
  exit /b 2
)

%PYTHON_EXE% -m scripts.kb_health
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo ============================================================
if "%EXIT_CODE%"=="0" (
  echo Knowledge Base health check: HEALTHY
) else (
  echo Knowledge Base health check: REVIEW NEEDED
  echo Review logs\kb_health\kb_health_*.txt
)
echo ============================================================
echo.
pause
exit /b %EXIT_CODE%
