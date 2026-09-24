@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title DocuBot - Production Environment Setup
chcp 65001 >nul

echo ============================================================
echo DocuBot v6.4.83 - Production Environment Setup
echo ============================================================
echo This one-click setup will:
echo   1. Use/install Python 3.11
echo   2. Create/reuse project venv
echo   3. Install requirements.txt
echo   4. Use/install Ollama
echo   5. Ensure qwen2.5:7b
echo   6. Ensure qwen3-embedding:8b
echo   7. Download/verify BAAI/bge-reranker-v2-m3
echo   8. Run a read-only production KB health check
echo.
echo It will NOT rebuild the knowledge base.
echo ============================================================
echo.

set "PY311="

if exist "%CD%\venv\Scripts\python.exe" (
  "%CD%\venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1
  if not errorlevel 1 (
    set "PY311=%CD%\venv\Scripts\python.exe"
    echo [OK] Existing project venv uses Python 3.11.
    goto :HAVE_VENV
  ) else (
    echo [FAIL] Existing venv is not Python 3.11.
    echo Production baseline is Python 3.11. Do not overwrite this venv automatically.
    echo Rename/remove the existing venv only after backing it up, then rerun this setup.
    pause
    exit /b 3
  )
)

for /f "usebackq delims=" %%I in (`py -3.11 -c "import sys; print(sys.executable)" 2^>nul`) do set "PY311=%%I"
if not defined PY311 if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set "PY311=%LocalAppData%\Programs\Python\Python311\python.exe"

if not defined PY311 (
  where winget >nul 2>&1
  if errorlevel 1 (
    echo [FAIL] Python 3.11 is missing and WinGet is unavailable.
    echo Install Python 3.11, then rerun this BAT.
    pause
    exit /b 2
  )

  echo [SETUP] Installing Python 3.11 with WinGet...
  winget install --exact --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
  if errorlevel 1 (
    echo [WARN] WinGet returned an error; checking whether Python 3.11 became available...
  )

  for /f "usebackq delims=" %%I in (`py -3.11 -c "import sys; print(sys.executable)" 2^>nul`) do set "PY311=%%I"
  if not defined PY311 if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set "PY311=%LocalAppData%\Programs\Python\Python311\python.exe"
)

if not defined PY311 (
  echo [FAIL] Python 3.11 could not be located after installation.
  echo Close this window, open a new terminal, and rerun this BAT.
  pause
  exit /b 2
)

echo [SETUP] Creating project venv with:
echo         %PY311%
"%PY311%" -m venv "%CD%\venv"
if errorlevel 1 (
  echo [FAIL] Could not create project venv.
  pause
  exit /b 1
)
set "PY311=%CD%\venv\Scripts\python.exe"

:HAVE_VENV
echo.
echo [SETUP] Updating pip tooling...
"%PY311%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo [FAIL] pip bootstrap failed.
  pause
  exit /b 1
)

echo.
echo [SETUP] Installing DocuBot Python requirements...
"%PY311%" -m pip install -r "%CD%\requirements.txt"
if errorlevel 1 (
  echo [FAIL] requirements.txt installation failed.
  pause
  exit /b 1
)

echo.
echo [SETUP] Installing/verifying external production models and runtime...
"%PY311%" -X utf8 -u -m scripts.setup_production_environment
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
  echo ============================================================
  echo [PASS] DocuBot production environment is ready.
  echo Normal use: open the project in VS Code and run run.py
  echo ============================================================
) else (
  echo ============================================================
  echo [FAIL] Environment setup did not complete cleanly.
  echo Check logs\environment_setup\ for the exact diagnostic.
  echo ============================================================
)

echo.
pause
exit /b %RC%
