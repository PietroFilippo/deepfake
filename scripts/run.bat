@echo off
cd /d "%~dp0\.."
call venv\Scripts\activate

if "%~1"=="" (
    echo Usage: run.bat [path_to_source_image]
    echo Example: run.bat images/my_photo.jpg
    pause
    exit /b
)

echo Starting application...
python main.py --source "%~1"
pause
