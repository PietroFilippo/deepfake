@echo off
cd /d "%~dp0\.."
call venv\Scripts\activate

if "%~1"=="" (
    echo Uso: run.bat [caminho_para_imagem_origem]
    echo Exemplo: run.bat images/minha_foto.jpg
    pause
    exit /b
)

echo Iniciando aplicação...
python main.py --source "%~1"
pause
