import time
import numpy as np
import onnxruntime
import os
import sys

# Add project root to path so src can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import setup_dll_directories, get_default_providers

# Set up DLL directories for Windows
setup_dll_directories()

model_path = "models/inswapper_128_fp16.onnx"
providers = get_default_providers()


print(f"Loading model: {model_path}")
sess = onnxruntime.InferenceSession(model_path, providers=providers)
print("Model loaded.")

# Inputs
# inswapper takes: 
# - target: [1, 3, 128, 128] (float32)
# - source_embedding: [1, 512] (float32)
target_img = np.random.randn(1, 3, 128, 128).astype(np.float32)
source_emb = np.random.randn(1, 512).astype(np.float32)

input_name_target = sess.get_inputs()[0].name
input_name_source = sess.get_inputs()[1].name

print("Warming up...")
for _ in range(5):
    sess.run(None, {input_name_target: target_img, input_name_source: source_emb})

print("Running benchmark (50 iterations)")
t0 = time.time()
for _ in range(50):
    sess.run(None, {input_name_target: target_img, input_name_source: source_emb})
t1 = time.time()

avg_time = (t1 - t0) / 50 * 1000
print(f"Average Inference Time: {avg_time:.2f} ms")
print(f"Estimated FPS: {1000/avg_time:.2f}")
