@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "PYTHON_EXE="
if exist "venv\Scripts\python.exe" set "PYTHON_EXE=venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "env\Scripts\python.exe" set "PYTHON_EXE=env\Scripts\python.exe"
if not defined PYTHON_EXE (
  echo [FAIL] DocuBot project Python environment not found.
  echo Expected venv\Scripts\python.exe or .venv\Scripts\python.exe.
  pause
  exit /b 2
)

"%PYTHON_EXE%" -X utf8 -u -m scripts.evaluate_v6_5_1_mixed_goal_benchmark
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [PASS] Mixed benchmark evaluation passed.
) else (
  echo [REVIEW] Mixed benchmark has one or more failed/missing goals. Review the generated evaluation files.
)
pause
exit /b %RC%
