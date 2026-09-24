@echo off
setlocal
cd /d "%~dp0"
set "PYEXE="
if exist "%~dp0..\venv\Scripts\python.exe" set "PYEXE=%~dp0..\venv\Scripts\python.exe"
if not defined PYEXE if exist "%~dp0..\.venv\Scripts\python.exe" set "PYEXE=%~dp0..\.venv\Scripts\python.exe"
if not defined PYEXE if exist "%~dp0..\env\Scripts\python.exe" set "PYEXE=%~dp0..\env\Scripts\python.exe"
if not defined PYEXE set "PYEXE=python"
"%PYEXE%" -X utf8 -u "%~dp0apply_v6_5_5_2_kb_notification_chat_disable_ux.py" "%~dp0.."
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [FAIL] Installer exit code: %RC%
if "%RC%"=="0" echo [PASS] DocuBot v6.5.5.2 notification/chat-disable UX hotfix completed.
echo.
pause
exit /b %RC%
