@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0artifacts\start_windows.ps1" %*
