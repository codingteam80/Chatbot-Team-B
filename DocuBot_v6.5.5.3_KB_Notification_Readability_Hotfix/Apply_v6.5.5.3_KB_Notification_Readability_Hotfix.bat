@echo off
setlocal
cd /d "%~dp0"
set "PYEXE="
if exist "%~dp0..\venv\Scripts\python.exe" set "PYEXE=%~dp0..\venv\Scripts\python.exe"
if not defined PYEXE if exist "%~dp0..\.venv\Scripts\python.exe" set "PYEXE=%~dp0..\.venv\Scripts\python.exe"
if not defined PYEXE if exist "%~dp0..\env\Scripts\python.exe" set "PYEXE=%~dp0..\env\Scripts\python.exe"
if not defined PYEXE set "PYEXE=python"
"%PYEXE%" -X utf8 -u "%~dp0apply_v6_5_5_3_kb_notification_readability.py" "%~dp0.."
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (
  echo [FAILED] v6.5.5.3 was not installed. Review the messages above.
) else (
  echo [PASS] v6.5.5.3 installed. Restart DocuBot and visually verify the success toast.
)
pause
exit /b %RC%
