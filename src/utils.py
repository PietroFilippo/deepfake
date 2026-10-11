"""
Shared utilities for the deepfake project.
Centralizes common functionality used across multiple modules.
"""
import os
import shutil
import sys

import cv2
import numpy as np

# Absolute paths so the app works no matter which directory it is started from.
# In the packaged app (PyInstaller) the folders live next to the executable.
if getattr(sys, 'frozen', False):
    PROJECT_ROOT = os.path.dirname(os.path.abspath(sys.executable))
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
TRT_CACHE_DIR = os.path.join(PROJECT_ROOT, 'trt_cache')


def model_path(filename):
    """Returns the absolute path of a file inside the models/ folder."""
    return os.path.join(MODELS_DIR, filename)


def read_image(path):
    """
    cv2.imread that also works with non-ASCII paths on Windows (e.g. accented file names).

    Returns:
        numpy.ndarray or None: BGR image, or None if the file could not be read
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, img):
    """cv2.imwrite that also works with non-ASCII paths on Windows."""
    ext = os.path.splitext(path)[1] or '.jpg'
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Could not encode image as {ext}")
    buf.tofile(path)


def ffmpeg_executable():
    """ffmpeg on the PATH, or the copy bundled with the imageio-ffmpeg package. None if neither exists."""
    path = shutil.which('ffmpeg')
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def ffprobe_executable():
    """ffprobe on the PATH (imageio-ffmpeg doesn't include it). None if missing."""
    return shutil.which('ffprobe')


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
    if getattr(sys, 'frozen', False):
        return  # The packaged app ships its own CUDA/cuDNN/TensorRT DLLs next to the executable
    
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


def get_default_providers(fp16=True):
    """
    Returns the default ONNX Runtime execution provider configuration.
    Prioritizes TensorRT > CUDA > CPU.

    Args:
        fp16 (bool): Let TensorRT build FP16 engines (faster, slightly less precise)

    Returns:
        list: List of providers with their settings
    """
    os.makedirs(TRT_CACHE_DIR, exist_ok=True)
    return [
        ('TensorrtExecutionProvider', {
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': TRT_CACHE_DIR,
            'trt_fp16_enable': fp16,
        }),
        ('CUDAExecutionProvider', {
            'cudnn_conv_algo_search': 'HEURISTIC',
            'arena_extend_strategy': 'kSameAsRequested',
        }),
        'CPUExecutionProvider'
    ]
