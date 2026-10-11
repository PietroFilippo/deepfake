"""
Shared utilities for the deepfake project.
Centralizes common functionality used across multiple modules.
"""
import os


def find_tensorrt_lib_path():
    """
    Automatically detects the TensorRT library path from an environment variable or common locations.
    
    Returns:
        str or None: Path to the TensorRT lib directory if found, None otherwise
    """
    trt_lib = None
    if 'TENSORRT_DIR' in os.environ:
        trt_lib = os.path.join(os.environ['TENSORRT_DIR'], 'lib')
    else:
        # Try common paths (Windows)
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
    Adds the required DLL directories to the Windows search path.
    This includes the ONNX Runtime, TensorRT and PyTorch library paths.
    
    Must be called at module initialization, before loading models.
    """
    if os.name != 'nt':
        return  # Only needed on Windows
    
    try:
        # Add the ONNX Runtime capi directory
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
        
        # Add the TensorRT lib directory
        trt_lib = find_tensorrt_lib_path()
        if trt_lib:
            os.add_dll_directory(trt_lib)
        
        # Add the PyTorch lib directory (for zlibwapi.dll and other dependencies)
        try:
            import torch
            torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
            if os.path.exists(torch_lib):
                os.add_dll_directory(torch_lib)
        except ImportError:
            pass
        
        print("DLL directories added to the search path.")
    except Exception as e:
        print(f"Warning: Failed to add DLL directories: {e}")


def get_default_providers():
    """
    Returns the default ONNX Runtime execution provider configuration.
    Prioritizes TensorRT > CUDA > CPU.
    
    Returns:
        list: List of providers with their settings
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
