import os
import shutil
import glob
import sys

# Add project root to path so src can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import find_tensorrt_lib_path

# Find TensorRT path using shared utility
trt_lib_path = find_tensorrt_lib_path()


if not trt_lib_path or not os.path.exists(trt_lib_path):
    print("Error: TensorRT not found. Set TENSORRT_DIR or install it in a default location.")
    exit(1)

# Find ONNX Runtime path in the current venv
try:
    import onnxruntime
    ort_base = os.path.dirname(onnxruntime.__file__)
    ort_capi_path = os.path.join(ort_base, 'capi')
except ImportError:
    print("Error: ONNX Runtime is not installed.")
    exit(1)

print(f"Copying DLLs from {trt_lib_path} to {ort_capi_path}.")

if not os.path.exists(trt_lib_path):
    print(f"Error: Source path {trt_lib_path} does not exist.")
    exit(1)

if not os.path.exists(ort_capi_path):
    print(f"Error: Destination path {ort_capi_path} does not exist.")
    exit(1)

dll_files = glob.glob(os.path.join(trt_lib_path, "*.dll"))
print(f"Found {len(dll_files)} DLLs.")

failed = 0
for dll in dll_files:
    filename = os.path.basename(dll)
    dest = os.path.join(ort_capi_path, filename)
    try:
        shutil.copy2(dll, dest)
        print(f"Copied {filename}")
    except Exception as e:
        print(f"Failed to copy {filename}: {e}")
        failed += 1

if failed:
    print(f"Done with {failed} failed copies")
    sys.exit(1)
print("Done")
