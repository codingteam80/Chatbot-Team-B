@echo off
setlocal
cd /d "%~dp0"
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PY=%ROOT%\venv\Scripts\python.exe"

echo ========================================================================
echo DocuBot FINAL CERTIFICATION v6.5.5.6
echo ========================================================================
echo This folder is intentionally kept separate from the production code.
echo It validates:
echo   1. Retrieval Recall / Precision / same-meaning consistency
echo   2. Answer generation from verified retrieval context
echo   3. Normal single-query steady-state latency target ^<= 25 sec
echo   4. Actual ADD / MODIFY / DELETE KB lifecycle and preservation
echo.
echo Ambiguous MultiQuery latency is measured and reported separately.
echo Startup prewarm cost is recorded separately and is not counted as a normal question.
echo.
echo IMPORTANT: Close DocuBot / Streamlit before this run so local Qdrant is not locked.
echo.

if not exist "%PY%" (
  echo [FAIL] Live DocuBot Python was not found:
  echo        %PY%
  echo.
  echo Place this entire folder directly inside:
  echo        C:\user_dev\company-chatbot
  echo then run this BAT from inside the folder.
  echo.
  pause
  exit /b 1
)

if not exist "%ROOT%\scripts\test_incremental_kb_lifecycle_live.py" (
  echo [FAIL] The v6.5.5.5 quality-test support files are missing from the live project.
  echo        Expected: %ROOT%\scripts\test_incremental_kb_lifecycle_live.py
  echo.
  pause
  exit /b 1
)

echo [PASS] Project root : %ROOT%
echo [PASS] Python       : %PY%
echo [PASS] QA folder    : %~dp0
echo.
echo Starting final certification...
echo ========================================================================
"%PY%" -X utf8 "%~dp0run_final_certification.py"
set "RC=%ERRORLEVEL%"
echo ========================================================================
if "%RC%"=="0" (
  echo [PASS] Final certification completed successfully.
) else (
  echo [CHECK] Final certification returned exit code %RC%.
  echo         The evidence ZIP contains the exact failing stage.
)
echo.
echo Send me the newest two files from:
echo   %ROOT%\logs\v6_5_5_final_certification
echo.
echo   DocuBot_v6.5.5.6_Final_Certification_Result_*.zip
echo   DocuBot_v6.5.5.6_Final_Certification_Result_*.zip.sha256.txt
echo.
pause
exit /b %RC%
