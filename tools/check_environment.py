import os
import sys
import ctypes

# Add project root to path so src can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import find_tensorrt_lib_path

def print_section(title):
    # Print a formatted section title
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check_python_info():
    # Check Python info
    print_section("PYTHON INFO")
    print(f"Python: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")

def check_tensorrt():
    # Check TensorRT installation
    print_section("TENSORRT")
    try:
        import tensorrt as trt
        print(f"TensorRT version: {trt.__version__}")
        return True
    except ImportError as e:
        print(f"TensorRT not found: {e}")
        return False

def check_onnxruntime():
    # Check ONNX Runtime installation
    print_section("ONNX RUNTIME")
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime version: {ort.__version__}")
        print(f"Available providers: {ort.get_available_providers()}")
        
        if 'TensorrtExecutionProvider' in ort.get_available_providers():
            print("TensorRT provider is available")
        else:
            print("TensorRT provider NOT available")
            
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            print("CUDA provider is available")
        else:
            print("CUDA provider NOT available - will use CPU only")
        return True
    except ImportError as e:
        print(f"Could not import onnxruntime: {e}")
        return False

def check_pytorch_cuda():
    # Check PyTorch and CUDA
    print_section("PYTORCH AND CUDA")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available in PyTorch: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version (PyTorch): {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available in PyTorch")
        return torch.cuda.is_available()
    except ImportError as e:
        print(f"Could not import torch: {e}")
        return False

def check_cuda_paths():
    # Check CUDA directories in PATH
    print_section("CUDA DIRECTORIES IN PATH")
    path_dirs = os.environ.get('PATH', '').split(';')
    cuda_paths = [p for p in path_dirs if 'cuda' in p.lower()]
    
    if cuda_paths:
        print("CUDA directories found in PATH:")
        for p in cuda_paths:
            print(f"  • {p}")
        return True
    else:
        print("No CUDA directories found in PATH")
        return False

def check_tensorrt_dlls():
    # Check TensorRT DLLs
    print_section("TENSORRT LIBRARIES")
    
    # Use shared utility to find the TensorRT path
    trt_lib_path = find_tensorrt_lib_path()
    
    if not trt_lib_path:
        print("TensorRT not found in default locations or TENSORRT_DIR")
        return False
        
    nvinfer_path = os.path.join(trt_lib_path, "nvinfer_10.dll")
    print(f"Checking for nvinfer_10.dll in: {trt_lib_path}")
    
    if not os.path.exists(trt_lib_path):
        print(f"TensorRT directory not found")
        return False
        
    if os.path.exists(nvinfer_path):
        print("nvinfer_10.dll exists")
        try:
            # Try loading it explicitly to see if any dependencies are missing
            ctypes.CDLL(nvinfer_path)
            print("nvinfer_10.dll loaded successfully")
            return True
        except Exception as e:
            print(f"Failed to load nvinfer_10.dll: {e}")
            print("This usually means a dependency (such as zlibwapi.dll or cudart) is missing from PATH")
            return False
    else:
        print("nvinfer_10.dll NOT found")
        return False

def check_dependencies():
    # Check important dependencies
    print_section("DEPENDENCIES")
    
    # Check for zlibwapi in PATH
    print("Checking for zlibwapi.dll in PATH.")
    found_zlib_in_path = False
    for p in os.environ['PATH'].split(';'):
        zlib_path = os.path.join(p, "zlibwapi.dll")
        if os.path.exists(zlib_path):
            print(f"zlibwapi.dll found in PATH: {zlib_path}")
            found_zlib_in_path = True
            break
    
    if not found_zlib_in_path:
        print("zlibwapi.dll NOT found in PATH")
    
    # Check for zlibwapi in known locations (torch/lib)
    print("Checking for zlibwapi.dll in known locations.")
    found_zlib_torch = False
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        zlib_torch_path = os.path.join(torch_lib, 'zlibwapi.dll')
        if os.path.exists(zlib_torch_path):
            print(f"zlibwapi.dll found in torch/lib: {zlib_torch_path}")
            found_zlib_torch = True
        else:
            print(f"zlibwapi.dll NOT found in: {torch_lib}")
    except ImportError:
        print("PyTorch is not installed, could not check torch/lib")
    
    # zlibwapi counts as OK if found in either location
    return found_zlib_in_path or found_zlib_torch

def main():
    # Run all checks
    print("\n" + "="*60)
    print("FULL ENVIRONMENT DIAGNOSTICS")
    print("="*60)
    
    results = {
        'Python': True,  # Always True if the script is running
        'TensorRT': check_tensorrt(),
        'ONNX Runtime': check_onnxruntime(),
        'PyTorch/CUDA': check_pytorch_cuda(),
        'CUDA PATH': check_cuda_paths(),
        'TensorRT DLLs': check_tensorrt_dlls(),
        'Dependencies': check_dependencies()
    }
    
    # Final summary
    print_section("SUMMARY")
    check_python_info()
    
    print("\nComponent status:")
    for component, status in results.items():
        status_icon = "✓" if status else "❌"
        print(f"{status_icon} {component}")
    
    # Overall result
    all_ok = all(results.values())
    
    print(f"\n{'='*60}")
    if all_ok:
        print("ENVIRONMENT CONFIGURED CORRECTLY.")
        print("All required components are available.")
    else:
        print("SOME CHECKS FAILED.")
        print("Review the errors above to fix the environment setup.")
    print(f"{'='*60}\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
