"""
Utilitários compartilhados para o projeto deepfake.
Centraliza funcionalidades comuns utilizadas em múltiplos módulos.
"""
import os


def find_tensorrt_lib_path():
    """
    Detecta automaticamente o caminho da biblioteca TensorRT a partir de variável de ambiente ou locais comuns.
    
    Returns:
        str or None: Caminho para o diretório lib do TensorRT se encontrado, None caso contrário
    """
    trt_lib = None
    if 'TENSORRT_DIR' in os.environ:
        trt_lib = os.path.join(os.environ['TENSORRT_DIR'], 'lib')
    else:
        # Tenta caminhos comuns (Windows)
        common_paths = [
            r"C:\Program Files\TensorRT-10.4.0.26\TensorRT-10.4.0.26\lib",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\lib",
            r"C:\TensorRT\lib",
        ]
        for path in common_paths:
            if os.path.exists(path):
                trt_lib = path
                break
    
    return trt_lib if trt_lib and os.path.exists(trt_lib) else None


def setup_dll_directories():
    """
    Adiciona diretórios DLL necessários ao caminho de busca para Windows.
    Isso inclui caminhos das bibliotecas ONNX Runtime, TensorRT e PyTorch.
    
    Deve ser chamado na inicialização do módulo antes de carregar modelos.
    """
    if os.name != 'nt':
        return  # Necessário apenas para Windows
    
    try:
        # Adiciona diretório capi do ONNX Runtime
        try:
            import insightface
            ort_capi = os.path.join(
                os.path.dirname(os.path.dirname(insightface.__file__)), 
                'onnxruntime', 
                'capi'
            )
            if os.path.exists(ort_capi):
                os.add_dll_directory(ort_capi)
        except ImportError:
            pass
        
        # Adiciona diretório lib do TensorRT
        trt_lib = find_tensorrt_lib_path()
        if trt_lib:
            os.add_dll_directory(trt_lib)
        
        # Adiciona diretório lib do PyTorch (para zlibwapi.dll e outras dependências)
        try:
            import torch
            torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
            if os.path.exists(torch_lib):
                os.add_dll_directory(torch_lib)
        except ImportError:
            pass
        
        print("Diretórios DLL adicionados ao caminho de busca.")
    except Exception as e:
        print(f"Aviso: Falha ao adicionar diretórios DLL: {e}")


def get_default_providers():
    """
    Retorna a configuração padrão dos provedores de execução do ONNX Runtime.
    Prioriza TensorRT > CUDA > CPU.
    
    Returns:
        list: Lista de provedores com suas configurações
    """
    return [
        ('TensorrtExecutionProvider', {
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': 'trt_cache',
            'trt_fp16_enable': True,
        }),
        ('CUDAExecutionProvider', {
            'cudnn_conv_algo_search': 'HEURISTIC',
            'arena_extend_strategy': 'kSameAsRequested',
        }),
        'CPUExecutionProvider'
    ]
