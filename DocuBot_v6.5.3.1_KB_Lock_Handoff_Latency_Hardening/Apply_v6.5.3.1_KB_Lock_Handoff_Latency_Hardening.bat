@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ==========================================================================================================
echo DocuBot v6.5.3.1 - KB Update Worker Lock Handoff + Latency Decision Diagnostics
echo ==========================================================================================================
echo [INFO] Close DocuBot/Streamlit before running this installer.
echo [INFO] No production Qdrant/BM25 rebuild is performed by this patch.
echo.

set "PYEXE="
for %%P in ("venv\Scripts\python.exe" ".venv\Scripts\python.exe" "env\Scripts\python.exe") do (
    if exist "..\%%~P" set "PYEXE=..\%%~P"
)
if not defined PYEXE set "PYEXE=python"

"%PYEXE%" -X utf8 -u apply_v6_5_3_1_kb_lock_handoff_latency.py
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo [PASS] v6.5.3.1 installer completed.
) else (
    echo [FAIL] v6.5.3.1 installer stopped with code %RC%.
)
echo Press any key to continue . . .
pause >nul
exit /b %RC%
