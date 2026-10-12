"""
Fast, non-blocking ONNX Runtime sessions with TensorRT.

Creating a TensorRT session holds Python's GIL for the whole time: several seconds when the engine
is cached, minutes when it has to be built. That freezes everything else in the process (the window,
the live preview), so on Windows the app was reported as "not responding" and closed.

Instead, each model's engine is built once in a separate process, which saves it with a small
"EP context" model (the original inputs/outputs plus a reference to the engine file). Loading that
model afterwards takes about half a second.
"""
import hashlib
import os
import shutil
import subprocess
import sys

import onnxruntime

from .utils import TRT_CACHE_DIR, PROJECT_ROOT, get_default_providers, setup_dll_directories

setup_dll_directories()
# Errors only (ONNX Runtime warns about every unused provider option and every TensorRT fallback)
onnxruntime.set_default_logger_severity(3)

CONTEXT_DIR = os.path.join(TRT_CACHE_DIR, 'context')
CONTEXT_FILE = 'model_ctx.onnx'


def tensorrt_available():
    return 'TensorrtExecutionProvider' in onnxruntime.get_available_providers()


def _context_dir(model_path, fp16):
    """Folder of the prebuilt engine for this model file, precision and ONNX Runtime version."""
    model_path = os.path.abspath(model_path)
    stat = os.stat(model_path)
    key = f"{os.path.normcase(model_path)}|{stat.st_size}|{stat.st_mtime_ns}|{fp16}|{onnxruntime.__version__}"
    digest = hashlib.sha1(key.encode('utf-8')).hexdigest()[:10]
    name = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(CONTEXT_DIR, f"{name}_{'fp16' if fp16 else 'fp32'}_{digest}")


def is_prepared(model_path, fp16=True):
    return not tensorrt_available() or os.path.exists(os.path.join(_context_dir(model_path, fp16), CONTEXT_FILE))


def _builder_command():
    """Command line that runs this app in engine-building mode (the packaged .exe or app.py)."""
    if getattr(sys, 'frozen', False):
        return [sys.executable]
    return [sys.executable, os.path.join(PROJECT_ROOT, 'app.py')]


def build_context_model(model_path, fp16=True):
    """Builds the TensorRT engine for `model_path` in a separate process (doesn't block this one)."""
    final_dir = _context_dir(model_path, fp16)
    work_dir = f"{final_dir}.building-{os.getpid()}"
    shutil.rmtree(work_dir, ignore_errors=True)
    cmd = _builder_command() + ['--build-engine', os.path.abspath(model_path), work_dir, '1' if fp16 else '0']
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    result = subprocess.run(cmd, capture_output=True, text=True, errors='replace', cwd=PROJECT_ROOT,
                            creationflags=flags)
    if result.returncode != 0 or not os.path.exists(os.path.join(work_dir, CONTEXT_FILE)):
        shutil.rmtree(work_dir, ignore_errors=True)
        output = (result.stdout + result.stderr).strip().splitlines()
        raise RuntimeError(f"Engine build failed for {os.path.basename(model_path)}: "
                           f"{output[-1] if output else 'exit code ' + str(result.returncode)}")
    if os.path.exists(final_dir):
        shutil.rmtree(work_dir, ignore_errors=True)  # built meanwhile by another call
    else:
        os.replace(work_dir, final_dir)


def create_session(model_path, fp16=True, on_status=None, providers=None):
    """
    InferenceSession for `model_path` on TensorRT (with CUDA/CPU fallback). The first time a model is
    used, its engine is built in a separate process (minutes); afterwards it loads in about 0.5 s.
    Explicit `providers` skip all that and load the model directly.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if providers is not None:
        return onnxruntime.InferenceSession(model_path, providers=providers)
    providers = get_default_providers(fp16)
    if not tensorrt_available():
        return onnxruntime.InferenceSession(model_path, providers=providers)

    name = os.path.basename(model_path)
    context_model = os.path.join(_context_dir(model_path, fp16), CONTEXT_FILE)
    if not os.path.exists(context_model):
        if on_status:
            on_status(f"Preparing {name} for your GPU (first time only, can take a few minutes)...")
        try:
            build_context_model(model_path, fp16)
        except Exception as e:
            print(f"[Engines] {e}. Loading {name} directly instead.")
            return onnxruntime.InferenceSession(model_path, providers=providers)

    try:
        return onnxruntime.InferenceSession(context_model, providers=providers)
    except Exception as e:
        # E.g. another GPU or driver: drop the prebuilt engine, it will be built again next time
        print(f"[Engines] Prebuilt engine for {name} could not be loaded ({e}); rebuilding next time.")
        shutil.rmtree(os.path.dirname(context_model), ignore_errors=True)
        return onnxruntime.InferenceSession(model_path, providers=providers)


def cached_array(model_path, name, compute):
    """`compute()`, saved next to the model's prebuilt engine so it isn't recomputed on every load."""
    import numpy as np
    folder = _context_dir(model_path, True)
    path = os.path.join(folder, f"{name}.npy")
    if os.path.exists(path):
        return np.load(path)
    value = compute()
    if os.path.isdir(folder):
        np.save(path, value)
    return value


# ---------------------------------------------------------------------- builder process

def _profile_options(model_path):
    """
    TensorRT only builds an engine up front when every input shape is known, so dynamic inputs get an
    explicit range: batch 1, and 320-640 for image sizes (the face detector's sizes).
    """
    import onnx
    graph = onnx.load(model_path, load_external_data=False).graph
    initializers = {i.name for i in graph.initializer}
    shapes = {'min': [], 'opt': [], 'max': []}
    dynamic = False
    for inp in graph.input:
        if inp.name in initializers:
            continue
        dims = [d.dim_value if d.dim_value > 0 else None for d in inp.type.tensor_type.shape.dim]
        dynamic |= None in dims
        for kind, image_size in (('min', 320), ('opt', 640), ('max', 640)):
            values = [d if d is not None else (image_size if i >= 2 else 1) for i, d in enumerate(dims)]
            shapes[kind].append(f"{inp.name}:{'x'.join(map(str, values))}")
    if not dynamic:
        return {}
    return {f'trt_profile_{kind}_shapes': ','.join(values) for kind, values in shapes.items()}


def build_engine_main(model_path, out_dir, fp16):
    """Entry point of the builder process (app.py --build-engine MODEL OUT_DIR FP16)."""
    import onnx
    os.makedirs(out_dir, exist_ok=True)
    context_model = os.path.join(out_dir, CONTEXT_FILE)
    options = {
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': out_dir,
        'trt_fp16_enable': fp16,
        'trt_dump_ep_context_model': True,
        'trt_ep_context_file_path': context_model,
        'trt_ep_context_embed_mode': 0,
    }
    options.update(_profile_options(model_path))
    onnxruntime.InferenceSession(model_path, providers=[('TensorrtExecutionProvider', options),
                                                        'CUDAExecutionProvider', 'CPUExecutionProvider'])
    if not os.path.exists(context_model):
        raise RuntimeError("TensorRT did not write an EP context model")

    fix_context_model(context_model, model_path)
    print(f"Built {context_model}")
    return 0


def fix_context_model(context_model, model_path):
    """
    Makes a dumped EP context model behave like the original model: engine path relative to the
    context model (required by ONNX Runtime), and the original inputs/outputs with their order and
    dynamic shapes (the dump sorts them and fixes the shapes to the build profile; insightface's
    detector relies on the output order).
    """
    import onnx
    model = onnx.load(context_model)
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.name == 'ep_cache_context':
                attribute.s = os.path.basename(attribute.s.decode('utf-8')).encode('utf-8')
    original = onnx.load(model_path, load_external_data=False).graph
    initializers = {i.name for i in original.initializer}
    del model.graph.input[:]
    model.graph.input.extend(v for v in original.input if v.name not in initializers)
    del model.graph.output[:]
    model.graph.output.extend(original.output)
    onnx.save(model, context_model)
