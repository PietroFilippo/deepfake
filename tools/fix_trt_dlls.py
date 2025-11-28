import os
import shutil
import glob
import sys

# Detecta caminho do TensorRT
trt_lib_path = None
if 'TENSORRT_DIR' in os.environ:
    trt_lib_path = os.path.join(os.environ['TENSORRT_DIR'], 'lib')
else:
    # Procura TensorRT em caminhos comuns (Windows, pode ser modificado)
    common_paths = [
        r"C:\Program Files\TensorRT-10.4.0.26\TensorRT-10.4.0.26\lib",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\lib",
        r"C:\TensorRT\lib",
    ]
    for path in common_paths:
        if os.path.exists(path):
            trt_lib_path = path
            break

if not trt_lib_path or not os.path.exists(trt_lib_path):
    print("Erro: TensorRT não encontrado. Configure TENSORRT_DIR ou instale em local padrão.")
    exit(1)

# Detecta caminho do ONNX Runtime no venv atual
try:
    import onnxruntime
    ort_base = os.path.dirname(onnxruntime.__file__)
    ort_capi_path = os.path.join(ort_base, 'capi')
except ImportError:
    print("Erro: ONNX Runtime não está instalado.")
    exit(1)

print(f"Copiando DLLs de {trt_lib_path} para {ort_capi_path}.")

if not os.path.exists(trt_lib_path):
    print(f"Erro: Caminho de origem {trt_lib_path} não existe.")
    exit(1)

if not os.path.exists(ort_capi_path):
    print(f"Erro: Caminho de destino {ort_capi_path} não existe.")
    exit(1)

dll_files = glob.glob(os.path.join(trt_lib_path, "*.dll"))
print(f"Encontradas {len(dll_files)} DLLs.")

for dll in dll_files:
    filename = os.path.basename(dll)
    dest = os.path.join(ort_capi_path, filename)
    try:
        shutil.copy2(dll, dest)
        print(f"Copiada {filename}")
    except Exception as e:
        print(f"Falha ao copiar {filename}: {e}")

print("Concluído")
