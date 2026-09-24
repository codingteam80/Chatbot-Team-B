@echo off
setlocal
cd /d "%~dp0"
title DocuBot Centralized LAN Server
chcp 65001 >nul

set "PYTHON_EXE="
if exist "venv\Scripts\python.exe" set "PYTHON_EXE=venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if not defined PYTHON_EXE (
  echo [FAIL] Project Python environment not found.
  echo Expected venv\Scripts\python.exe or .venv\Scripts\python.exe.
  pause
  exit /b 2
)

echo ============================================================================
echo DocuBot - Centralized Office LAN Server
echo ============================================================================
echo Server runtime: %PYTHON_EXE%
echo.
"%PYTHON_EXE%" -X utf8 -u -m scripts.run_lan_server
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo DocuBot LAN server exited with code %RC%.
pause
exit /b %RC%
