@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Configure_DocuBot_LAN_Firewall.ps1" -Port 8501
