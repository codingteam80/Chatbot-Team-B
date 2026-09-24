@echo off
setlocal
cd /d "%~dp0"
title DocuBot - Production Model Check
chcp 65001 >nul

echo ============================================================
echo DocuBot finalized production models
echo ============================================================
echo Generation : qwen2.5:7b
echo Embedding  : qwen3-embedding:8b
echo.

where ollama >nul 2>&1
if errorlevel 1 (
  echo [FAIL] Ollama command was not found.
  pause
  exit /b 2
)

ollama list | findstr /I /C:"qwen2.5:7b" >nul 2>&1
if errorlevel 1 (
  echo qwen2.5:7b is missing. Pulling it now...
  ollama pull qwen2.5:7b
  if errorlevel 1 exit /b 1
) else (
  echo [OK] qwen2.5:7b
)

ollama list | findstr /I /C:"qwen3-embedding:8b" >nul 2>&1
if errorlevel 1 (
  echo qwen3-embedding:8b is missing. Pulling it now...
  ollama pull qwen3-embedding:8b
  if errorlevel 1 exit /b 1
) else (
  echo [OK] qwen3-embedding:8b
)

echo.
echo Production Ollama models are ready.
pause
exit /b 0
