import os
import sys
import ctypes

def print_section(title):
    # Imprime um título de seção formatado
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check_python_info():
    # Verifica as informações do Python
    print_section("INFORMAÇÕES DO PYTHON")
    print(f"Python: {sys.version}")
    print(f"Executável Python: {sys.executable}")
    print(f"Diretório de trabalho: {os.getcwd()}")

def check_tensorrt():
    # Verifica a instalação do TensorRT
    print_section("TENSORRT")
    try:
        import tensorrt as trt
        print(f"TensorRT versão: {trt.__version__}")
        return True
    except ImportError as e:
        print(f"TensorRT não encontrado: {e}")
        return False

def check_onnxruntime():
    # Verifica a instalação do ONNX Runtime
    print_section("ONNX RUNTIME")
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime versão: {ort.__version__}")
        print(f"Providers disponíveis: {ort.get_available_providers()}")
        
        if 'TensorrtExecutionProvider' in ort.get_available_providers():
            print("TensorRT provider está disponível")
        else:
            print("TensorRT provider NÃO disponível")
            
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            print("CUDA provider está disponível")
        else:
            print("CUDA provider NÃO disponível - irá usar apenas CPU")
        return True
    except ImportError as e:
        print(f"Não foi possível importar onnxruntime: {e}")
        return False

def check_pytorch_cuda():
    # Verifica PyTorch e CUDA
    print_section("PYTORCH E CUDA")
    try:
        import torch
        print(f"Versão do PyTorch: {torch.__version__}")
        print(f"CUDA disponível no PyTorch: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Versão do CUDA (PyTorch): {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA não está disponível no PyTorch")
        return torch.cuda.is_available()
    except ImportError as e:
        print(f"Não foi possível importar torch: {e}")
        return False

def check_cuda_paths():
    # Verifica diretórios CUDA no PATH
    print_section("DIRETÓRIOS CUDA NO PATH")
    path_dirs = os.environ.get('PATH', '').split(';')
    cuda_paths = [p for p in path_dirs if 'cuda' in p.lower()]
    
    if cuda_paths:
        print("Diretórios CUDA encontrados no PATH:")
        for p in cuda_paths:
            print(f"  • {p}")
        return True
    else:
        print("Nenhum diretório CUDA encontrado no PATH")
        return False

def check_tensorrt_dlls():
    # Verifica DLLs do TensorRT
    print_section("BIBLIOTECAS TENSORRT")
    
    # Auto-detecção do TensorRT
    trt_lib_path = None
    if 'TENSORRT_DIR' in os.environ:
        trt_lib_path = os.path.join(os.environ['TENSORRT_DIR'], 'lib')
    else:
        common_paths = [
            r"C:\Program Files\TensorRT-10.4.0.26\TensorRT-10.4.0.26\lib",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\lib",
            r"C:\TensorRT\lib",
        ]
        for path in common_paths:
            if os.path.exists(path):
                trt_lib_path = path
                break
    
    if not trt_lib_path:
        print("TensorRT não encontrado em locais padrão ou TENSORRT_DIR")
        return False
        
    nvinfer_path = os.path.join(trt_lib_path, "nvinfer_10.dll")
    print(f"Verificando nvinfer_10.dll em: {trt_lib_path}")
    
    if not os.path.exists(trt_lib_path):
        print(f"Diretório TensorRT não encontrado")
        return False
        
    if os.path.exists(nvinfer_path):
        print("Arquivo nvinfer_10.dll existe")
        try:
            # Tenta carregar explicitamente para ver se faltam dependências
            ctypes.CDLL(nvinfer_path)
            print("nvinfer_10.dll carregada com sucesso")
            return True
        except Exception as e:
            print(f"Falha ao carregar nvinfer_10.dll: {e}")
            print("Isso geralmente significa que uma dependência (como zlibwapi.dll ou cudart) está faltando no PATH")
            return False
    else:
        print("Arquivo nvinfer_10.dll NÃO encontrado")
        return False

def check_dependencies():
    # Verifica dependências importantes
    print_section("DEPENDÊNCIAS")
    
    # Verifica zlibwapi no PATH
    print("Verificando zlibwapi.dll no PATH.")
    found_zlib_in_path = False
    for p in os.environ['PATH'].split(';'):
        zlib_path = os.path.join(p, "zlibwapi.dll")
        if os.path.exists(zlib_path):
            print(f"zlibwapi.dll encontrada no PATH: {zlib_path}")
            found_zlib_in_path = True
            break
    
    if not found_zlib_in_path:
        print("zlibwapi.dll NÃO encontrada no PATH")
    
    # Verifica zlibwapi em locais conhecidos (torch/lib)
    print("Verificando zlibwapi.dll em locais conhecidos.")
    found_zlib_torch = False
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        zlib_torch_path = os.path.join(torch_lib, 'zlibwapi.dll')
        if os.path.exists(zlib_torch_path):
            print(f"zlibwapi.dll encontrada em torch/lib: {zlib_torch_path}")
            found_zlib_torch = True
        else:
            print(f"zlibwapi.dll NÃO encontrada em: {torch_lib}")
    except ImportError:
        print("PyTorch não está instalado, não foi possível verificar torch/lib")
    
    # zlibwapi é considerado OK se estiver em qualquer um dos locais
    return found_zlib_in_path or found_zlib_torch

def main():
    # Executa todas as verificações
    print("\n" + "="*60)
    print("DIAGNÓSTICO COMPLETO DO AMBIENTE")
    print("="*60)
    
    results = {
        'Python': True,  # Sempre True se o script está rodando
        'TensorRT': check_tensorrt(),
        'ONNX Runtime': check_onnxruntime(),
        'PyTorch/CUDA': check_pytorch_cuda(),
        'CUDA PATH': check_cuda_paths(),
        'TensorRT DLLs': check_tensorrt_dlls(),
        'Dependências': check_dependencies()
    }
    
    # Resumo final
    print_section("RESUMO")
    check_python_info()
    
    print("\nStatus dos componentes:")
    for component, status in results.items():
        status_icon = "✓" if status else "❌"
        print(f"{status_icon} {component}")
    
    # Análise geral
    all_ok = all(results.values())
    
    print(f"\n{'='*60}")
    if all_ok:
        print("AMBIENTE CONFIGURADO CORRETAMENTE.")
        print("Todos os componentes necessários estão disponíveis.")
    else:
        print("ALGUMAS VERIFICAÇÕES FALHARAM.")
        print("Revise os erros acima para configurar o ambiente.")
    print(f"{'='*60}\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
