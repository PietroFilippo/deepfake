"""
Fast, non-blocking ONNX Runtime sessions with TensorRT.

Creating a TensorRT session holds Python's GIL for the whole time: several seconds when the engine
is cached, minutes when it has to be built. That freezes everything else in the process (the window,
the live preview), so on Windows the app was reported as "not responding" and closed.

Instead, each model's engine is built once in a separate process, which saves it with a small
"EP context" model (the original inputs/outputs plus a reference to the engine file). Loading that
model afterwards takes about half a second. TensorRT is never loaded in-process the slow way: if a
prebuilt engine can't be made or used, the model runs on CUDA instead.
"""
import collections
import hashlib
import os
import shutil
import subprocess
import sys
import threading
import traceback
import uuid

import onnxruntime

from .utils import TRT_CACHE_DIR, PROJECT_ROOT, get_default_providers, setup_dll_directories

setup_dll_directories()
# Errors only (ONNX Runtime warns about every unused provider option and every TensorRT fallback)
onnxruntime.set_default_logger_severity(3)

CONTEXT_DIR = os.path.join(TRT_CACHE_DIR, 'context')
CONTEXT_FILE = 'model_ctx.onnx'
UNSUPPORTED_FILE = 'not_fully_supported.txt'  # TensorRT can't run the whole model: it runs on CUDA

_build_locks = collections.defaultdict(threading.Lock)
_active_builds = set()
_active_lock = threading.Lock()
_cancelled = False


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


def _engine_files(context_model):
    import onnx
    model = onnx.load(context_model)
    return [attribute.s.decode('utf-8') for node in model.graph.node for attribute in node.attribute
            if attribute.name == 'ep_cache_context']


def _is_complete(folder):
    """A finished engine folder: the context model and every engine file it uses (or the unsupported mark)."""
    if os.path.exists(os.path.join(folder, UNSUPPORTED_FILE)):
        return True
    context_model = os.path.join(folder, CONTEXT_FILE)
    if not os.path.exists(context_model):
        return False
    try:
        return all(os.path.exists(os.path.join(folder, name)) for name in _engine_files(context_model))
    except Exception:
        return False


def is_prepared(model_path, fp16=True):
    """True if `model_path` loads quickly (no engine build needed)."""
    if not tensorrt_available():
        return True
    try:
        return _is_complete(_context_dir(model_path, fp16))
    except OSError:
        return True  # missing model: loading it reports the error


def _builder_command():
    """Command line that runs this app in engine-building mode (the packaged .exe or app.py)."""
    if getattr(sys, 'frozen', False):
        return [sys.executable]
    return [sys.executable, os.path.join(PROJECT_ROOT, 'app.py')]


def cancel_builds():
    """Stops running engine builds (when the app closes) and refuses new ones."""
    global _cancelled
    _cancelled = True
    with _active_lock:
        processes = list(_active_builds)
    for process in processes:
        process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def build_context_model(model_path, fp16=True):
    """Builds the TensorRT engine for `model_path` in a separate process (doesn't block this one)."""
    final_dir = _context_dir(model_path, fp16)
    with _build_locks[final_dir]:
        if _is_complete(final_dir):
            return
        if _cancelled:
            raise RuntimeError("cancelled")
        staging = f"{final_dir}.building-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        cmd = _builder_command() + ['--build-engine', os.path.abspath(model_path), staging, '1' if fp16 else '0']
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       errors='replace', cwd=PROJECT_ROOT, creationflags=flags)
            with _active_lock:
                _active_builds.add(process)
            try:
                output, _ = process.communicate()
            finally:
                with _active_lock:
                    _active_builds.discard(process)
            if _cancelled:
                raise RuntimeError("cancelled")
            if process.returncode != 0 or not _is_complete(staging):
                lines = output.strip().splitlines()
                raise RuntimeError(f"engine build failed for {os.path.basename(model_path)}: "
                                   f"{lines[-1] if lines else 'exit code ' + str(process.returncode)}")
            # Publish: replace an incomplete leftover, or keep a complete one made by another instance
            if not _is_complete(final_dir):
                shutil.rmtree(final_dir, ignore_errors=True)
                try:
                    os.replace(staging, final_dir)
                except OSError:
                    if not _is_complete(final_dir):
                        raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)


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
    if not tensorrt_available():
        return onnxruntime.InferenceSession(model_path, providers=get_default_providers(fp16))

    name = os.path.basename(model_path)
    folder = _context_dir(model_path, fp16)
    for attempt in range(2):
        if not _is_complete(folder):
            if on_status:
                reason = "first time only" if attempt == 0 else "the GPU, driver or TensorRT changed"
                on_status(f"Preparing {name} for your GPU ({reason}, can take a few minutes)...")
            try:
                build_context_model(model_path, fp16)
            except Exception as e:
                if _cancelled:
                    raise
                print(f"[Engines] {e}")
                break
        if os.path.exists(os.path.join(folder, UNSUPPORTED_FILE)):
            break
        try:
            return onnxruntime.InferenceSession(os.path.join(folder, CONTEXT_FILE),
                                                providers=get_default_providers(fp16))
        except Exception as e:
            # E.g. another GPU, driver or TensorRT version: build it again once
            print(f"[Engines] Prebuilt engine for {name} could not be loaded ({e}).")
            shutil.rmtree(folder, ignore_errors=True)

    # Never load TensorRT in-process the slow way (it would freeze the app): run on CUDA instead
    print(f"[Engines] Running {name} on CUDA without TensorRT.")
    return onnxruntime.InferenceSession(model_path, providers=get_default_providers(fp16)[1:])


def cached_array(model_path, name, compute):
    """`compute()`, saved next to the model's prebuilt engine so it isn't recomputed on every load."""
    import numpy as np
    try:
        path = os.path.join(_context_dir(model_path, True), f"{name}.npy")
        return np.load(path)
    except Exception:
        pass  # missing, or a damaged file: recompute
    value = compute()
    try:
        temp = f"{path}.{uuid.uuid4().hex[:8]}.tmp.npy"
        np.save(temp, value)
        os.replace(temp, path)
    except Exception:
        pass  # the cache is optional
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
    try:
        return _build_engine(model_path, out_dir, fp16)
    except Exception:
        # Report through the exit code and output: an uncaught exception in the packaged (windowless)
        # app would open an error dialog nobody sees, and the app would wait for it forever
        traceback.print_exc(file=sys.stdout)
        return 1


def _build_engine(model_path, out_dir, fp16):
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
    if not os.path.exists(context_model) or not fix_context_model(context_model, model_path):
        # TensorRT runs only part of the model: the app uses CUDA for it (remembered, so the build
        # isn't attempted on every start)
        for name in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, name))
        with open(os.path.join(out_dir, UNSUPPORTED_FILE), 'w', encoding='utf-8') as f:
            f.write("TensorRT can't run this whole model; it runs on CUDA.\n")
        print(f"{os.path.basename(model_path)}: not fully supported by TensorRT, using CUDA")
        return 0
    print(f"Built {context_model}")
    return 0


def fix_context_model(context_model, model_path):
    """
    Makes a dumped EP context model behave like the original model: engine path relative to the
    context model (required by ONNX Runtime), and the original inputs/outputs with their order and
    dynamic shapes (the dump sorts them and fixes the shapes to the build profile; insightface's
    detector relies on the output order).

    Returns False if TensorRT doesn't cover the whole model (the dump then isn't a usable replacement).
    """
    import onnx
    model = onnx.load(context_model)
    original = onnx.load(model_path, load_external_data=False).graph
    initializers = {i.name for i in original.initializer}
    inputs = [v for v in original.input if v.name not in initializers]
    produced = {name for node in model.graph.node for name in node.output}
    consumed = {name for node in model.graph.node for name in node.input if name}
    whole_model = (all(node.op_type == 'EPContext' for node in model.graph.node)
                   and consumed <= {v.name for v in inputs}
                   and {v.name for v in original.output} <= produced)
    if not whole_model:
        return False

    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.name == 'ep_cache_context':
                attribute.s = os.path.basename(attribute.s.decode('utf-8')).encode('utf-8')
    del model.graph.input[:]
    model.graph.input.extend(inputs)
    del model.graph.output[:]
    model.graph.output.extend(original.output)
    onnx.save(model, context_model)
    return True
