import onnx
from onnxconverter_common import float16

input_model_path = "models/inswapper_128.onnx"
output_model_path = "models/inswapper_128_fp16.onnx"

print(f"Loading model from {input_model_path}.")
try:
    model = onnx.load(input_model_path)
    print("Model loaded successfully.")

    print("Converting to FP16.")
    # keep_io_types=True keeps inputs/outputs as float32 where needed for compatibility
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    
    print(f"Saving FP16 model to {output_model_path}.")
    onnx.save(model_fp16, output_model_path)
    print("Conversion complete.")

except Exception as e:
    print(f"Error during conversion: {e}")
