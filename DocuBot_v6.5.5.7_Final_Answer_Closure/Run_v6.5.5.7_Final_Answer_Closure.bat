@echo off
setlocal
chcp 65001 >nul
set "HERE=%~dp0"
for %%I in ("%HERE%..") do set "ROOT=%%~fI"
set "PY=%ROOT%\venv\Scripts\python.exe"

echo ============================================================================
echo DocuBot v6.5.5.7 - FINAL ANSWER CLOSURE + CHUNKING EVIDENCE
echo ============================================================================
echo.
echo This run will:
echo   1. back up the current answer-service file
echo   2. apply the generic verified-answer contract closure
echo   3. run deterministic answer-contract checks
echo   4. generate READ-ONLY chunking evidence from the active KB
echo   5. retest LLM answers using the already-verified retrieval contexts
echo.
echo IMPORTANT: Close DocuBot / Streamlit first.
echo.

if not exist "%ROOT%\services\answer_service.py" (
  echo [FAIL] Put this whole folder directly inside the live project:
  echo        C:\user_dev\company-chatbot\DocuBot_v6.5.5.7_Final_Answer_Closure
  echo.
  pause
  exit /b 2
)

if not exist "%PY%" (
  echo [FAIL] Live Python not found: %PY%
  echo.
  pause
  exit /b 2
)

echo [PASS] Project root : %ROOT%
echo [PASS] Python       : %PY%
echo.
"%PY%" "%HERE%apply_test_and_evidence.py"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [PASS] v6.5.5.7 closure finished successfully.
  echo Send the newest Result ZIP + SHA256 from:
  echo   %ROOT%\logs\v6_5_5_7_closure
) else (
  echo [FAIL] Closure did not certify. The runner restored the prior production files.
  echo Send the newest Result ZIP + SHA256 from:
  echo   %ROOT%\logs\v6_5_5_7_closure
)
echo.
pause
exit /b %RC%
