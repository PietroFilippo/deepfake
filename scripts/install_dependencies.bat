@echo off
echo Instalando PyTorch com suporte CUDA...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo.
echo Instalando demais dependencias...
pip install -r ../requirements.txt

echo.
echo Instalacao concluida!
pause
