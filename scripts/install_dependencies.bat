@echo off
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo.
echo Installing remaining dependencies...
pip install -r ../requirements.txt

echo.
echo Installation complete!
pause
