@echo off
rem Starts the Deepfake app (no console window; messages go to deepfake.log)
cd /d "%~dp0"
if not exist "venv\Scripts\pythonw.exe" (
    echo venv not found. Follow the installation steps in README.md first.
    pause
    exit /b 1
)
start "" "venv\Scripts\pythonw.exe" app.py
