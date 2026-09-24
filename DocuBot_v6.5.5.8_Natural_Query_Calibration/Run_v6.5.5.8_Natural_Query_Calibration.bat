@echo off
setlocal
cd /d "%~dp0"
set "ROOT=%~dp0.."
set "PY=%ROOT%\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================================
echo DocuBot v6.5.5.8 Natural Query Calibration
echo ============================================================
echo Project root: %ROOT%
echo.
"%PY%" "%~dp0apply_validate_and_package.py"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo PASS. Open DocuBot and run CALIBRATION_SET_A_CRITICAL_RETEST.txt
) else (
  echo FAILED. Previous production files were restored when possible.
)
echo.
pause
exit /b %RC%
