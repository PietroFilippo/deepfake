import time
import numpy as np
import onnxruntime
import os
import insightface

# Configura DLLs (mesmo que swapper.py)
if os.name == 'nt':
    try:
        ort_capi = os.path.join(os.path.dirname(os.path.dirname(insightface.__file__)), 'onnxruntime', 'capi')
        if os.path.exists(ort_capi): 
            os.add_dll_directory(ort_capi)
        
        # Auto-detecção do TensorRT
        trt_lib = None
        if 'TENSORRT_DIR' in os.environ:
            trt_lib = os.path.join(os.environ['TENSORRT_DIR'], 'lib')
        else:
            # Procura TensorRT em caminhos comuns (Windows, pode ser modificado)
            common_paths = [
                r"C:\Program Files\TensorRT-10.4.0.26\TensorRT-10.4.0.26\lib",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\lib",
                r"C:\TensorRT\lib",
            ]
            for path in common_paths:
                if os.path.exists(path):
                    trt_lib = path
                    break
        
        if trt_lib and os.path.exists(trt_lib): 
            os.add_dll_directory(trt_lib)
        
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.exists(torch_lib): 
            os.add_dll_directory(torch_lib)
    except: 
        pass

model_path = "models/inswapper_128_fp16.onnx"
providers = [
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

print(f"Carregando modelo: {model_path}")
sess = onnxruntime.InferenceSession(model_path, providers=providers)
print("Modelo carregado.")

# Entradas
# inswapper recebe: 
# - target: [1, 3, 128, 128] (float32)
# - source_embedding: [1, 512] (float32)
target_img = np.random.randn(1, 3, 128, 128).astype(np.float32)
source_emb = np.random.randn(1, 512).astype(np.float32)

input_name_target = sess.get_inputs()[0].name
input_name_source = sess.get_inputs()[1].name

print("Aquecendo...")
for _ in range(5):
    sess.run(None, {input_name_target: target_img, input_name_source: source_emb})

print("Fazendo benchmark (50 iterações)")
t0 = time.time()
for _ in range(50):
    sess.run(None, {input_name_target: target_img, input_name_source: source_emb})
t1 = time.time()

avg_time = (t1 - t0) / 50 * 1000
print(f"Tempo Médio de Inferência: {avg_time:.2f} ms")
print(f"FPS Estimado: {1000/avg_time:.2f}")
