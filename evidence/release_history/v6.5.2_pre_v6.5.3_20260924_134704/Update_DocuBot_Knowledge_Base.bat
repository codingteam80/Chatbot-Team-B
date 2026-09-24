@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title DocuBot - Safe Technical Knowledge Base Update
chcp 65001 >nul

echo ============================================================
echo DocuBot - Safe Technical Knowledge Base Update
echo ============================================================
echo [INFO] Source root: data\technical_documents
echo [INFO] The previous working Qdrant/BM25 index is preserved on failure.
echo.

set "PYTHON_EXE="
if exist "venv\Scripts\python.exe" set "PYTHON_EXE=venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"

if not defined PYTHON_EXE (
  echo [FAIL] Project Python environment not found.
  echo [INFO] Run Setup_DocuBot_Production_Environment.bat first.
  pause
  exit /b 2
)

"%PYTHON_EXE%" -X utf8 -u -m scripts.update_server_kb
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
  echo [PASS] Knowledge base update workflow completed.
) else (
  echo [FAIL] Knowledge base update did not complete.
  echo [INFO] The previous working index remains protected.
  echo [INFO] Review logs\kb_update\ and logs\server_kb_update\ if needed.
)
echo.
pause
exit /b %RC%
