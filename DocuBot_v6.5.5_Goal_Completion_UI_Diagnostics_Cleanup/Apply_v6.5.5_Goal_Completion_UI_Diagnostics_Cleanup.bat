@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ================================================================================
echo DocuBot v6.5.5 - Goal Completion UI + Diagnostics + Safe Cleanup
echo ================================================================================
echo.

echo [INFO] Stop DocuBot/Streamlit before applying this patch.
echo [INFO] This installer verifies the exact v6.5.4 baseline before changing files.
echo.

set "PATCH_DIR=%~dp0"
set "PROJECT="

for %%D in ("%PATCH_DIR%.." "%PATCH_DIR%..\company-chatbot" "%CD%" "%CD%\company-chatbot") do (
    if not defined PROJECT (
        if exist "%%~fD\run.py" if exist "%%~fD\config\settings.py" set "PROJECT=%%~fD"
    )
)

if not defined PROJECT (
    echo [FAIL] Could not locate the DocuBot project near this patch folder.
    echo        Extract this patch inside or next to the company-chatbot project folder.
    goto :fail
)

echo [INFO] Project: %PROJECT%

set "PYTHON_EXE="
for %%P in ("%PROJECT%\venv\Scripts\python.exe" "%PROJECT%\.venv\Scripts\python.exe" "%PROJECT%\env\Scripts\python.exe") do (
    if not defined PYTHON_EXE if exist "%%~fP" set "PYTHON_EXE=%%~fP"
)

if not defined PYTHON_EXE (
    echo [FAIL] Could not find project Python in venv, .venv, or env.
    goto :fail
)

echo [INFO] Python: %PYTHON_EXE%
echo.
"%PYTHON_EXE%" -X utf8 -u "%PATCH_DIR%apply_v6_5_5_goal_completion.py" "%PROJECT%"
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo [PASS] v6.5.5 installer finished successfully.
    echo [INFO] Send the generated Result ZIP and .sha256.txt before running the runtime completion suite.
    echo.
    pause
    exit /b 0
)

echo [FAIL] v6.5.5 installer stopped with exit code %RC%.
echo [INFO] Review the console. The installer is fail-closed and rolls back after a post-copy validation failure.
:fail
echo.
pause
exit /b 1
