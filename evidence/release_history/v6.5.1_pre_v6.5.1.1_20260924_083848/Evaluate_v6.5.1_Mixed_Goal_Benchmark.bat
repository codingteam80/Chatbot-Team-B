@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -X utf8 -m scripts.evaluate_v6_5_1_mixed_goal_benchmark
) else (
  python -X utf8 -m scripts.evaluate_v6_5_1_mixed_goal_benchmark
)

set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [PASS] Mixed benchmark evaluation passed.
) else (
  echo [REVIEW] Mixed benchmark has one or more failed/missing goals. Review the generated evaluation files.
)
pause
exit /b %RC%
