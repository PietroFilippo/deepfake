# Deepfake em Tempo Real com Webcam

Aplicação de face swapping em tempo real usando webcam, com aceleração GPU via TensorRT e CUDA.

## Características

- **Performance otimizada**: Usa TensorRT para inferência acelerada em GPU
- **Processamento assíncrono**: Pipeline multi-threaded com buffer de frames
- **Detecção inteligente**: Face detection com downscaling periódico
- **Suporte a múltiplos providers**: TensorRT → CUDA → CPU (fallback automático)

## Pré-requisitos

### Hardware
- GPU NVIDIA com suporte CUDA
- Webcam

### Software
- **Python 3.11**
- **CUDA 12.4+** ([Download](https://developer.nvidia.com/cuda-downloads))
- **TensorRT 10.4.0** ([Download](https://developer.nvidia.com/tensorrt)) - Requer login NVIDIA
  - Extrair para um dos locais padrão:
    - `C:\Program Files\TensorRT-10.4.0.26\` (recomendado)
    - `C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\`
    - `C:\TensorRT\`
  - Ou defina variável de ambiente: `TENSORRT_DIR=<seu_caminho_tensorrt>`

## Instalação

### 1. Clone o repositório
```bash
git clone <seu-repositorio>
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

### 4. Baixe o modelo inswapper
Coloque o arquivo `inswapper_128.onnx` ou a versão 16fp `inswapper_128_fp16.onnx` caso já tenha na pasta `models/`.

### 5. Verifique o ambiente
```bash
python tools/check_environment.py
```

Você deve ver: **AMBIENTE CONFIGURADO CORRETAMENTE**

## Estrutura do Projeto

```
deepfake/
├── src/                  # Módulos principais
│   ├── camera.py          # Captura de webcam com threading
│   └── swapper.py         # Face detection e swapping
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
├── main.py              # Aplicação principal
└── requirements.txt     # Dependências Python
```

## Uso

### Execução básica
```bash
python main.py --source images/minha_foto.jpg
```

### Usando script auxiliar
```bash
scripts\run.bat images/minha_foto.jpg
```

### Opções
```bash
python main.py --source <imagem> [--model <caminho_modelo>]
```

- `--source`: Imagem do rosto que será aplicado (obrigatório)
- `--model`: Caminho para modelo inswapper (padrão: `models/inswapper_128_fp16.onnx`)

### Controles
- **q**: Sair da aplicação

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

## Configuração Avançada

### Ajustar número de workers
Edite `src/swapper.py`, linha 89:
```python
self.max_workers = min(cpu_count, 5)  # Altere 5 para seu valor
```

### Ajustar resolução de detecção
Edite `src/swapper.py`, linha 46:
```python
def __init__(self, model_path, providers=None, det_size=(320, 320)):
```
Valores menores = mais rápido, menos preciso.

### Ajustar intervalo de detecção
Edite `main.py`, linha 50:
```python
future = swapper.process_frame_async(frame)  # detect_interval padrão: 5
```

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

Com RTX 4060 Ti + TensorRT FP16:
- **Detecção**: ~15-20ms por frame (em 50% downscale)
- **Swapping**: ~5-10ms por rosto
- **FPS alvo**: 15-30 FPS (depende de número de rostos e configurações)

## Notas

- Primeira execução é lenta (TensorRT compila engines e cria cache em `trt_cache/`)
- Execuções subsequentes são muito mais rápidas
- Cache TensorRT é específico para GPU
- Modelos FP16 são ~2x mais rápidos que FP32
