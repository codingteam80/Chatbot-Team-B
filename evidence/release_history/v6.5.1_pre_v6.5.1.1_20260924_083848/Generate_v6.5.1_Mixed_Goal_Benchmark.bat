@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -X utf8 -m scripts.generate_v6_5_1_mixed_goal_benchmark
) else (
  python -X utf8 -m scripts.generate_v6_5_1_mixed_goal_benchmark
)

if errorlevel 1 (
  echo.
  echo [FAIL] Benchmark sheet generation failed.
  pause
  exit /b 1
)

echo.
echo [PASS] Benchmark sheet created under logs\mixed_goal_benchmark.
pause
