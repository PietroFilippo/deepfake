# Deepfake em Tempo Real com Webcam e em Vídeos/Imagens

Aplicação de face swapping em tempo real usando webcam, em vídeos e em imagens estáticas, com aceleração GPU via TensorRT e CUDA.

## Características

- **Performance otimizada**: Usa TensorRT para inferência acelerada em GPU
- **Processamento assíncrono**: Pipeline multi-threaded com buffer de frames
- **Detecção inteligente**: Face detection com downscaling periódico
- **Suporte a múltiplos providers**: TensorRT → CUDA → CPU (fallback automático)
- **Múltiplos modos de entrada**: Webcam em tempo real, vídeos, GIFs ou imagens estáticas
- **Preservação de áudio**: Mantém áudio original ao processar vídeos
- **Sistema de gravação**: Grave a saída processada em tempo real

## Pré-requisitos

### Hardware
- GPU NVIDIA com suporte CUDA
- Webcam (opcional)
- Microfone para gravação de áudio (opcional)

### Software
- **Python 3.11**
- **FFmpeg** (Adicionado ao PATH do sistema) - Essencial para áudio ([Download](https://www.ffmpeg.org/download.html))
- **CUDA 12.4+** (Adicionado ao PATH do sistema) ([Download](https://developer.nvidia.com/cuda-downloads))
- **TensorRT 10.4.0** (Adicionado ao PATH do sistema) ([Download](https://developer.nvidia.com/tensorrt)) - Requer login NVIDIA
  - Extrair para um dos locais padrão:
    - `C:\Program Files\TensorRT-10.4.0.26\` (recomendado)
    - `C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\`
    - `C:\TensorRT\`
  - Ou defina variável de ambiente: `TENSORRT_DIR=<seu_caminho_tensorrt>`

## Instalação

### 1. Clone o repositório
```bash
git clone <repositorio>
cd deepfake
```

### 2. Crie o ambiente virtual
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instale as dependências
```bash
scripts\install_dependencies.bat
```

Ou manualmente:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 4. Instale o TensorRT após de ter baixado e extraido a pasta
```bash
pip install "C:\Program Files\TensorRT-10.4.0.26\python\tensorrt_lean-10.4.0-cp311-none-win_amd64.whl"
```

### 5. Baixe os modelos necessários por fora
Coloque os arquivos `GFPGANv1.4.onnx` e `inswapper_128.onnx` ou sua versão 16fp `inswapper_128_fp16.onnx` caso já tenha na pasta `models/`.

### 6. Verifique o ambiente
```bash
python tools/check_environment.py
```

Você deve ver: **AMBIENTE CONFIGURADO CORRETAMENTE**

## Estrutura do Projeto

```
deepfake/
├── src/                  # Módulos principais
│   ├── camera.py          # Captura de webcam/vídeo com threading
│   ├── swapper.py         # Face detection e swapping
│   └── utils.py           # Utilitários compartilhados (DLL setup, providers)
├── tools/                # Utilitários Python
│   ├── check_environment.py   # Diagnóstico completo
│   ├── benchmark_model.py     # Benchmark de inferência
│   ├── convert_fp16_v2.py     # Conversão para FP16
│   └── fix_trt_dlls.py        # Copia DLLs do TensorRT
├── scripts/              # Scripts de automação
│   ├── install_dependencies.bat
│   └── run.bat
├── models/               # Modelos ONNX (inswapper)
├── images/               # Imagens de origem
├── outputs/              # Vídeos e imagens processados
├── main.py              # Aplicação principal
└── requirements.txt     # Dependências Python
```

## Uso

### Modos de Operação

#### 1. Webcam em Tempo Real (Padrão)
```bash
python main.py --source images/minha_foto.jpg
```

#### 2. Processamento de Vídeo (Offline)
Processa um vídeo frame-by-frame e salva com áudio preservado:
```bash
python main.py --source images/minha_foto.jpg --video images/meu_video.mp4
```
- Processamento offline para qualidade máxima
- Áudio original preservado (via MoviePy/FFmpeg)
- Barra de progresso em tempo real
- Salva automaticamente em `outputs/` com nome único

#### 3. Processamento de Imagem Estática
Troca rosto em uma imagem:
```bash
python main.py --source images/minha_foto.jpg --image images/alvo.jpg
```
- Processa uma única imagem
- Salva automaticamente em `outputs/`
- Exibe resultado na tela
- Salva automaticamente em `outputs/`

#### 4. Processamento de GIF
Troca rosto em um arquivo GIF animado:
```bash
python main.py --source images/minha_foto.jpg --gif images/animacao.gif
```
- Processa frame a frame
- Mantém a velocidade (FPS) original
- Salva como novo GIF em `outputs/`

### Usando script auxiliar
```bash
scripts\run.bat images/minha_foto.jpg
```

### Opções
```bash
python main.py --source <imagem> [opções]
```

#### Argumentos Principais
- `--source`: Imagem do rosto que será aplicado (obrigatório)
- `--model`: Caminho para modelo inswapper (padrão: `models/inswapper_128_fp16.onnx`)
- `--video`: Caminho para vídeo de entrada (processamento offline)
- `--image`: Caminho para imagem de entrada (processamento estático)
- `--gif`: Caminho para GIF de entrada (processamento animado)
- `--out`: Caminho customizado para arquivo de saída
- `--enhance`: Ativa GFPGAN por padrão ao iniciar (Pode ser usado com vídeos e imagens)

#### Argumentos de Performance
- `--max-workers`: Número de threads paralelas (Padrão: auto).
  - Menor (2-3) = Menor latência (menos delay), FPS menor.
  - Maior (5-6) = Maior FPS, maior latência (mais delay).
- `--detect-interval`: Intervalo de quadros para detecção de rosto (Padrão: 5).
  - Aumentar (ex: 10) reduz uso de CPU e pode aumentar FPS da GPU.
- `--camera-fps`: Solicita FPS específico para a webcam (Padrão: 30).
- `--virtual-cam`: Ativa saída para OBS Virtual Camera (Útil para Discord, Zoom, etc).

### Controles (Modo Webcam)
- **q**: Sair da aplicação
- **n**: Trocar para próxima imagem da pasta `images/`
- **p**: Trocar para imagem anterior da pasta `images/`
- **x**: Ativar/Desativar troca de rosto (mostra webcam original)
- **e**: Ativar/Desativar "melhoria" de rosto (GFPGAN)
- **r**: Iniciar/Parar gravação (salva em `outputs/`)
- **u**: Mostrar/Ocultar interface (FPS, status, indicador REC)

## "Melhoria" de Rosto (Face Enhancer)

O projeto suporta melhoria de rosto usando o modelo **GFPGAN** para corrigir rostos borrados em resoluções maiores. Isso aumenta a resolução, mas deixa os rostos mais "plásticos". Qualidade bem pior no escuro, mas pode ser melhor de perto, varia de acordo com a qualidade da webcam.

- **Modelo**: `models/GFPGANv1.4.onnx` (Baixado externamente igual os outros modelos)
- **Ativação**: Pressione a tecla `e` durante a execução.
- **Performance**: Esse enhancer roda em **FP32** (Full Precision) no TensorRT para evitar artefatos visuais. Isso consome mais recursos que o swapper (FP16), então espere uma queda de FPS quando ativado. 

## Câmera Virtual (Discord/Zoom/Teams)

Para usar o deepfake em chamadas de vídeo:

1.  Certifique-se de ter o **OBS Studio** instalado (ele instala o driver da câmera virtual).
2.  Execute com a flag `--virtual-cam`:
    ```bash
    python main.py --source images/face.jpg --max-workers 4 --detect-interval 5 --virtual-cam
    ```
3.  No Discord/Zoom/Teams, selecione **OBS Virtual Camera** como sua câmera.
    
> **Nota:** O script suporta 1080p (1920x1080). Se desejar alterar a resolução padrão, edite `src/camera.py` (linhas 12 e 13) e `main.py` (linha 42).

## Gravação e Saída

### Sistema de Gravação
No modo webcam, você pode gravar a saída processada:
1. Pressione **'r'** para iniciar a gravação
2. Um indicador vermelho "REC" aparecerá na tela
3. Pressione **'r'** novamente para parar
4. O áudio do microfone será gravado e sincronizado automaticamente
5. O vídeo é salvo em `outputs/output_<timestamp>.mp4` (com áudio)

### Arquivos de Saída
Todos os arquivos processados são salvos em `outputs/` com nomes únicos:
- **Vídeos**: `processed_<video>_with_<source>_<timestamp>.mp4`
- **Imagens**: `processed_<imagem>_with_<source>_<timestamp>.jpg`
- **Gravações**: `output_<timestamp>.mp4`

### Preservação de Áudio
Ao processar vídeos com `--video`, o áudio original é preservado automaticamente:
- Requer `moviepy` instalado
- Usa codec AAC para áudio de alta qualidade

## Otimização de Performance

Encontre o equilíbrio ideal para seu hardware:

### 1. Modo Baixa Latência (Recomendado pra interação)
Prioriza resposta instantânea. FPS será menor (~12-15), mas o vídeo "acompanha" seus movimentos.
```bash
python main.py --source images/c.jpg --max-workers 2
```

### 2. Modo Alto FPS (Recomendado pra gravação)
Prioriza fluidez. FPS será maior (~25-30), mas haverá um pequeno atraso (delay).
```bash
python main.py --source images/c.jpg --max-workers 5 --camera-fps 60
```

### 3. Modo Equilibrado
Um meio termo entre fluidez e resposta.
```bash
python main.py --source images/c.jpg --max-workers 3 --detect-interval 8
```

### 4. Modo que Funcionou Bem pra Mim
```bash
python main.py --source images/c.jpg --max-workers 4 --detect-interval 5
```

## Scripts Utilitários

### Verificação de ambiente
```bash
python tools/check_environment.py
```
Verifica TensorRT, CUDA, ONNX Runtime, PyTorch e todas as dependências.

### Benchmark de modelo
```bash
python tools/benchmark_model.py
```
Mede tempo médio de inferência e FPS estimado.

### Conversão para FP16
```bash
python tools/convert_fp16_v2.py
```
Converte modelos ONNX de FP32 para FP16 (~2x mais rápido).

### Fix de DLLs do TensorRT
```bash
python tools/fix_trt_dlls.py
```
Copia DLLs do TensorRT para ONNX Runtime (raramente necessário com auto-detecção).

## Troubleshooting

### Erro: `nvinfer_10.dll not found`
**Opção 1 - Instalação em local padrão:**
Instale TensorRT em um dos locais padrão:
- `C:\Program Files\TensorRT-10.4.0.26\TensorRT-10.4.0.26\`
- `C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\`
- `C:\TensorRT\`

**Opção 2 - Local customizado:**
Defina a variável de ambiente `TENSORRT_DIR`:
```bash
setx TENSORRT_DIR "C:\seu\caminho\TensorRT-10.4.0.26"
```

**Opção 3 - Fix manual:**
1. Execute: `python tools/fix_trt_dlls.py`
2. Ou adicione ao PATH: `C:\seu\caminho\TensorRT\lib`

### Erro: `zlibwapi.dll not found`
O `zlibwapi.dll` vem com PyTorch. Reinstale:
```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### FPS baixo
1. Use modelo FP16: `python tools/convert_fp16_v2.py`
2. Reduza resolução de detecção: `det_size=(256, 256)` em `src/swapper.py`
3. Aumente intervalo de detecção: `detect_interval=10` em `main.py`
4. Verifique se TensorRT está sendo usado: `python tools/check_environment.py`

### Providers não disponíveis
Execute `python tools/check_environment.py` para diagnóstico completo.
- Se falta `CUDAExecutionProvider`: Reinstale `onnxruntime-gpu`
- Se falta `TensorrtExecutionProvider`: Verifique instalação do TensorRT ou configure `TENSORRT_DIR`

## Performance Esperada

Testes feitos com RTX 4060 Ti + TensorRT FP16:
- **Detecção**: ~15-20ms por frame (em 50% downscale)
- **Swapping**: ~5-10ms por rosto
- **FPS alvo**: 15-30 FPS (depende de número de rostos e configurações)

## Notas

- Primeira execução é lenta (TensorRT compila engines e cria cache em `trt_cache/`)
- Execuções subsequentes são muito mais rápidas
- Cache TensorRT é específico para GPU
- Modelos FP16 são ~2x mais rápidos que FP32
