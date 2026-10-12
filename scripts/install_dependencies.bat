@echo off
rem Installs the Python dependencies into the active environment (works from any folder)
cd /d "%~dp0\.."
echo Installing PyTorch with CUDA support...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo.
echo Installing remaining dependencies...
python -m pip install -r requirements.txt

echo.
echo Installation complete!
pause
