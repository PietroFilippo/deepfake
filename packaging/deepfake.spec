# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the desktop app: one folder (dist\Deepfake\Deepfake.exe + _internal) that runs
# ONNX Runtime with TensorRT and CUDA without a CUDA/cuDNN/TensorRT install (only the NVIDIA driver).
# Build with scripts\build_exe.bat (it runs packaging\build.py, which cleans PATH first).
import glob
import importlib.util
import os
import sys

from PyInstaller.utils.hooks import collect_data_files

REPO = os.path.dirname(SPECPATH)
sys.path.insert(0, REPO)
from src.utils import find_tensorrt_lib_path  # noqa: E402

# GPU runtime DLLs: CUDA 12.4 runtime + cuDNN 9 from the torch wheel (the same DLLs the dev environment
# loads, so TensorRT engine caches stay compatible) and TensorRT 10 from its install folder.
CUDA_SRC = os.environ.get("PKG_CUDA_SRC") or os.path.join(
    importlib.util.find_spec("torch").submodule_search_locations[0], "lib")
TRT_SRC = os.environ.get("PKG_TRT_SRC") or find_tensorrt_lib_path()
if not TRT_SRC:
    raise SystemExit("TensorRT not found: set TENSORRT_DIR (see README.md)")

CUDA_DLLS = ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cufft64_11.dll"]
CUDNN_DLLS = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CUDA_SRC, "cudnn*64_9.dll")))
TRT_DLLS = ["nvinfer_10.dll", "nvinfer_plugin_10.dll", "nvonnxparser_10.dll", "nvinfer_builder_resource_10.dll"]

# They must sit at the top level of _internal, which the bootloader puts on the DLL search path:
# nvinfer/cudnn load their sub-libraries by bare name (in onnxruntime/capi TensorRT crashed the process).
gpu_binaries = ([(os.path.join(CUDA_SRC, n), ".") for n in CUDA_DLLS + CUDNN_DLLS]
                + [(os.path.join(TRT_SRC, n), ".") for n in TRT_DLLS])
for src, _ in gpu_binaries:
    if not os.path.isfile(src):
        raise SystemExit(f"Missing GPU DLL: {src}")

a = Analysis(
    [os.path.join(REPO, "app.py")],
    pathex=[REPO],
    binaries=gpu_binaries,
    datas=collect_data_files("insightface"),
    hiddenimports=["src.utils", "src.gui.main_window"],
    runtime_hooks=[os.path.join(SPECPATH, "rthook_env.py")],
    # torch is only a DLL source at build time; the app never imports it when frozen
    excludes=["torch", "torchvision", "torchaudio", "tensorrt_lean", "tensorrt"],
    noarchive=False,
)
# The dependency scan adds second copies of the GPU DLLs under their source-relative paths
# (torch/lib/..., ~1 GB of duplicates): keep only the top-level copies.
GPU_NAMES = {n.lower() for n in CUDA_DLLS + CUDNN_DLLS + TRT_DLLS}
a.binaries = [e for e in a.binaries
              if not (os.path.basename(e[0]).lower() in GPU_NAMES and os.path.dirname(e[0]) not in ("", "."))]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Deepfake",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Deepfake",
)
