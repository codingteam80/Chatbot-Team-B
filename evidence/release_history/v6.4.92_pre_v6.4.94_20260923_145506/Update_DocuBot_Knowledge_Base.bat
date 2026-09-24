@echo off
setlocal
cd /d "%~dp0"
title DocuBot Server Knowledge Base Update
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
"%PYTHON_EXE%" -X utf8 -u -m scripts.update_server_kb
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%
