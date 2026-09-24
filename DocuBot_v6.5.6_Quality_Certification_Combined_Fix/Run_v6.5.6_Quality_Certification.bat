@echo off
setlocal EnableExtensions
chcp 65001 >nul
title DocuBot v6.5.6 - Quality Certification

set "LIVE_ROOT=C:\user_dev\company-chatbot"
set "RUN_ROOT=%~dp0"
if "%RUN_ROOT:~-1%"=="\" set "RUN_ROOT=%RUN_ROOT:~0,-1%"
set "PYTHON_EXE="

rem Prefer the folder where this BAT is located, but only when it is a complete live project.
if exist "%RUN_ROOT%\scripts\run_v6_5_6_quality_certification.py" (
  if exist "%RUN_ROOT%\venv\Scripts\python.exe" set "PYTHON_EXE=%RUN_ROOT%\venv\Scripts\python.exe"
  if not defined PYTHON_EXE if exist "%RUN_ROOT%\.venv\Scripts\python.exe" set "PYTHON_EXE=%RUN_ROOT%\.venv\Scripts\python.exe"
  if not defined PYTHON_EXE if exist "%RUN_ROOT%\env\Scripts\python.exe" set "PYTHON_EXE=%RUN_ROOT%\env\Scripts\python.exe"
)

rem Known live office project fallback. This is the path used by the last successful KB update.
if not defined PYTHON_EXE if exist "%LIVE_ROOT%\scripts\run_v6_5_6_quality_certification.py" if exist "%LIVE_ROOT%\venv\Scripts\python.exe" (
  set "RUN_ROOT=%LIVE_ROOT%"
  set "PYTHON_EXE=%LIVE_ROOT%\venv\Scripts\python.exe"
)

if not defined PYTHON_EXE (
  echo ============================================================
  echo DocuBot v6.5.6 - QUALITY CERTIFICATION
  echo ============================================================
  echo [FAIL] The certification runner is not in the live DocuBot project,
  echo        or the live project's Python environment is missing.
  echo.
  if exist "%LIVE_ROOT%\venv\Scripts\python.exe" (
    echo [FOUND] Live Python environment:
    echo         %LIVE_ROOT%\venv\Scripts\python.exe
    echo.
    if not exist "%LIVE_ROOT%\scripts\run_v6_5_6_quality_certification.py" (
      echo [ACTION] The v6.5.6 patch files have NOT been copied into the live project root.
      echo          Copy the CONTENTS of DocuBot_v6.5.6_Quality_Certification_Exact_Patch.zip
      echo          directly into:
      echo          %LIVE_ROOT%
      echo          Choose Replace when Windows asks.
      echo.
      echo          Then run:
      echo          %LIVE_ROOT%\Run_v6.5.6_Quality_Certification.bat
    ) else (
      echo [ACTION] The v6.5.6 files exist in the live project, but no supported
      echo          project Python could be selected. Verify the venv folder.
    )
  ) else (
    echo [NOT FOUND] %LIVE_ROOT%\venv\Scripts\python.exe
    echo [ACTION] Verify that you are using the original working project at:
    echo          %LIVE_ROOT%
    echo          Do NOT run the certification from a newly extracted full-project folder.
  )
  echo.
  pause
  exit /b 2
)

cd /d "%RUN_ROOT%"

if not exist "app.py" (
  echo [FAIL] Project root sanity check failed: app.py not found in %CD%
  pause
  exit /b 3
)
if not exist "storage" (
  echo [FAIL] Project root sanity check failed: storage folder not found in %CD%
  pause
  exit /b 3
)
if not exist "data" (
  echo [FAIL] Project root sanity check failed: data folder not found in %CD%
  pause
  exit /b 3
)

echo ============================================================
echo DocuBot v6.5.6 - QUALITY CERTIFICATION
echo ============================================================
echo [INFO] Project root : %CD%
echo [INFO] Python       : %PYTHON_EXE%
echo.
echo This controlled test will:
echo   1. validate runtime preservation and KB Health
echo   2. create a temporary probe, index it, modify it, delete it,
echo      and verify deletion from manifest/Qdrant/BM25
echo   3. run same-meaning Recall/Precision/semantic retrieval tests
echo   4. run answer generation only after retrieval passes
echo   5. verify normal-question combined latency against 25 seconds
echo.
echo IMPORTANT: Close the DocuBot Streamlit/server process before this certification
 echo so the local Qdrant files are not held open by another process.
echo The live delete test refuses to start unless the KB plan is NOOP.
echo It includes fail-safe cleanup/recovery for its temporary probe file.
echo.

"%PYTHON_EXE%" -X utf8 -u -m scripts.run_v6_5_6_quality_certification
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
  echo [PASS] v6.5.6 quality certification completed successfully.
) else (
  echo [REVIEW] One or more certification gates did not pass.
)
echo [INFO] Send back the newest files from:
echo        logs\v6_5_6_quality_certification\
echo        DocuBot_v6.5.6_Quality_Certification_Result_*.zip
echo        DocuBot_v6.5.6_Quality_Certification_Result_*.zip.sha256.txt
echo.
pause
exit /b %RC%
