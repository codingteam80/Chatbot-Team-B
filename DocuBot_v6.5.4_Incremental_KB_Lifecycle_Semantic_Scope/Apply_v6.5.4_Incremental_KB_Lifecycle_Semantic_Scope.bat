@echo off
setlocal
cd /d "%~dp0"

set "ROOT=%~dp0.."
if exist "%ROOT%\run.py" goto :run
set "ROOT=%~dp0..\company-chatbot"
if exist "%ROOT%\run.py" goto :run
set "ROOT=%~dp0..\.."
if exist "%ROOT%\run.py" goto :run
set "ROOT=%~dp0..\..\company-chatbot"
if exist "%ROOT%\run.py" goto :run

echo [FAIL] DocuBot project not found near this patch folder.
echo Extract this patch beside the company-chatbot project, then run again.
pause
exit /b 2

:run
set "PY=%ROOT%\venv\Scripts\python.exe"
if exist "%PY%" goto :apply
set "PY=%ROOT%\.venv\Scripts\python.exe"
if exist "%PY%" goto :apply
set "PY=%ROOT%\env\Scripts\python.exe"
if exist "%PY%" goto :apply

echo [FAIL] Project Python not found under venv, .venv, or env.
pause
exit /b 3

:apply
"%PY%" -X utf8 -u "%~dp0apply_v6_5_4_incremental_kb_semantic_scope.py" "%ROOT%"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [PASS] v6.5.4 patch completed.
) else (
  echo [FAIL] v6.5.4 patch stopped with exit code %RC%.
)
pause
exit /b %RC%
