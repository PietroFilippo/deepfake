import argparse
import time
import numpy as np
import onnxruntime
import os
import sys

# Add project root to path so src can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import setup_dll_directories, get_default_providers, model_path as default_model_path

# Set up DLL directories for Windows
setup_dll_directories()

parser = argparse.ArgumentParser(description="Benchmark the inference time of a swap model")
parser.add_argument("--model", default=default_model_path("inswapper_128_fp16.onnx"),
                    help="Path to the ONNX model (e.g. models/hyperswap_1a_256.onnx)")
args = parser.parse_args()

model_path = args.model
providers = get_default_providers()


print(f"Loading model: {model_path}")
sess = onnxruntime.InferenceSession(model_path, providers=providers)
print("Model loaded.")

# Inputs
# Swap models take:
# - target: [1, 3, size, size] (float32), 128 for inswapper, 256 for HyperSwap
# - source: [1, 512] (float32)
# Random data with each input's own shape (dynamic dimensions become 1)
feeds = {i.name: np.random.randn(*[d if isinstance(d, int) else 1 for d in i.shape]).astype(np.float32)
         for i in sess.get_inputs()}
print("Inputs: " + ", ".join(f"{name} {list(value.shape)}" for name, value in feeds.items()))

print("Warming up...")
for _ in range(5):
    sess.run(None, feeds)

print("Running benchmark (50 iterations)")
t0 = time.time()
for _ in range(50):
    sess.run(None, feeds)
t1 = time.time()

avg_time = (t1 - t0) / 50 * 1000
print(f"Average Inference Time: {avg_time:.2f} ms")
print(f"Estimated FPS: {1000/avg_time:.2f}")
