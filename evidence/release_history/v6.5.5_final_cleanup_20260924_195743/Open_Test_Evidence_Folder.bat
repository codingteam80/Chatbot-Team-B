@echo off
setlocal
cd /d "%~dp0"
if not exist "logs\test_evidence" mkdir "logs\test_evidence"
start "" explorer "%CD%\logs\test_evidence"
exit /b 0
