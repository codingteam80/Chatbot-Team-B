# ============================================================
# DocuBot Streamlit Runner with UTF-8 Logging
# ============================================================

# Always run from the project folder where this script is located.
Set-Location $PSScriptRoot

# Use UTF-8 for Windows console input/output.
$utf8 = New-Object System.Text.UTF8Encoding($false)

[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

# Force Python to use UTF-8 for stdout, stderr, files, and subprocess output.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# Switch the active Windows console code page to UTF-8.
chcp 65001 | Out-Null

# Create the log folder.
$logDirectory = Join-Path $PSScriptRoot "logs"

New-Item `
    -ItemType Directory `
    -Path $logDirectory `
    -Force `
    | Out-Null

# Create a timestamped UTF-8 log file.
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"

$logFile = Join-Path `
    $logDirectory `
    "docubot_$timestamp.log"

[System.IO.File]::WriteAllText(
    $logFile,
    "",
    $utf8
)

Write-Host "Starting DocuBot..."
Write-Host "Log file: $logFile"
Write-Host ""

# Run Python explicitly in UTF-8 mode.
# Write every line both to the terminal and to the UTF-8 log file.
& python -X utf8 -u -m streamlit run app.py 2>&1 |
    ForEach-Object {

        $line = $_.ToString()

        [Console]::Out.WriteLine(
            $line
        )

        [System.IO.File]::AppendAllText(
            $logFile,
            $line + [Environment]::NewLine,
            $utf8
        )
    }

$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host "DocuBot stopped with exit code: $exitCode"

exit $exitCode
