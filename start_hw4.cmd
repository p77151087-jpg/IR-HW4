@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please run scripts\setup_hw4.ps1 first.
    pause
    exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_hw4.ps1" -OpenBrowser
if errorlevel 1 pause
