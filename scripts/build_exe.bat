@echo off
rem Builds dist\Deepfake\Deepfake.exe: one folder (~4 GB, includes the CUDA/cuDNN/TensorRT DLLs)
cd /d "%~dp0\.."
venv\Scripts\python.exe -m pip install --quiet pyinstaller
venv\Scripts\python.exe packaging\build.py
pause
