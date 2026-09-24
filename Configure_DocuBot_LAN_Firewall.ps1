param([int]$Port = 8501)

$ErrorActionPreference = "Stop"
$ruleName = "DocuBot LAN Server TCP $Port"

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "Administrator permission is required. Opening UAC prompt..."
    $args = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Port $Port"
    Start-Process powershell.exe -Verb RunAs -ArgumentList $args
    exit 0
}

Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue |
    Remove-NetFirewallRule -ErrorAction SilentlyContinue

New-NetFirewallRule `
    -DisplayName $ruleName `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -LocalPort $Port `
    -Profile Private `
    -RemoteAddress LocalSubnet | Out-Null

Write-Host ""
Write-Host "Firewall rule created: $ruleName" -ForegroundColor Green
Write-Host "Allowed source          : LocalSubnet only"
Write-Host "Windows network profile : Private only"
Write-Host "Public Internet exposure: NOT configured"
Write-Host ""

$profiles = Get-NetConnectionProfile -ErrorAction SilentlyContinue
foreach ($profile in $profiles) {
    Write-Host ("Network: {0} | Profile: {1}" -f $profile.Name, $profile.NetworkCategory)
}
Write-Host ""
Write-Host "If the office LAN is marked Public, change that trusted office network to Private before using DocuBot LAN access."
Read-Host "Press Enter to close"
