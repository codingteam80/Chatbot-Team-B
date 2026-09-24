@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "ROOT=%CD%"
set "PY=%ROOT%\venv\Scripts\python.exe"

cls
echo ========================================================================
echo DocuBot FINAL QUALITY CHECK v6.5.5.5
echo ========================================================================
echo This single run checks:
echo   1. Retrieval Recall / Precision / semantic consistency
echo   2. Answer generation only after verified retrieval
echo   3. Normal-question combined latency target ^<= 25 seconds
echo   4. Actual transactional ADD / MODIFY / DELETE lifecycle
echo      including deletion from Manifest + Qdrant + BM25 and KB preservation
echo.
echo IMPORTANT: Close DocuBot / Streamlit first so local Qdrant is not locked.
echo.

if not exist "%PY%" (
    echo [FAIL] Live DocuBot Python was not found:
    echo        %PY%
    echo.
    echo Run this BAT only from the live project root:
    echo        C:\user_dev\company-chatbot
    echo.
    pause
    exit /b 2
)

if not exist "%ROOT%\scripts\run_v6_5_5_completion_suite.py" (
    echo [FAIL] Required final-quality scripts are missing from this project copy.
    echo Apply the exact patch contents directly to:
    echo        C:\user_dev\company-chatbot
    echo.
    pause
    exit /b 3
)

set "PYTHONPATH=%ROOT%;%PYTHONPATH%"
echo [PASS] Project root : %ROOT%
echo [PASS] Python       : %PY%
echo.
echo Starting the full final-quality run now...
echo ========================================================================
echo.

"%PY%" -X utf8 "%ROOT%\scripts\run_v6_5_5_completion_suite.py" --with-live-kb-lifecycle
set "RC=%ERRORLEVEL%"

echo.
echo ========================================================================
if "%RC%"=="0" (
    echo [PASS] Final quality runner completed successfully.
) else (
    echo [CHECK] Final quality runner returned exit code %RC%.
    echo         This normally means one of the measured goals FAILED and the
    echo         evidence ZIP contains the exact failing stage.
)
echo.
echo Send me the newest two files from:
echo   %ROOT%\logs\v6_5_5_completion
echo.
echo   DocuBot_v6.5.5.5_Final_Quality_Result_*.zip
echo   DocuBot_v6.5.5.5_Final_Quality_Result_*.zip.sha256.txt
echo ========================================================================
pause
exit /b %RC%
