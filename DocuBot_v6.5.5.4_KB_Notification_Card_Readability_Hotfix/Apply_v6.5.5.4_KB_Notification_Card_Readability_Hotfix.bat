@echo off
setlocal
cd /d "%~dp0"
set "PYEXE="
if exist "%~dp0..\venv\Scripts\python.exe" set "PYEXE=%~dp0..\venv\Scripts\python.exe"
if not defined PYEXE if exist "%~dp0..\.venv\Scripts\python.exe" set "PYEXE=%~dp0..\.venv\Scripts\python.exe"
if not defined PYEXE if exist "%~dp0..\env\Scripts\python.exe" set "PYEXE=%~dp0..\env\Scripts\python.exe"
if not defined PYEXE set "PYEXE=python"
"%PYEXE%" -X utf8 -u "%~dp0apply_v6_5_5_4_kb_notification_card.py" "%~dp0.."
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (
  echo [FAILED] v6.5.5.4 was not installed. Review the messages above.
) else (
  echo [PASS] v6.5.5.4 installed. Restart DocuBot and visually verify the success notification card.
)
pause
exit /b %RC%
