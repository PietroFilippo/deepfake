import onnxruntime
import argparse
import sys

def inspect_model(model_path):
    try:
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    with open("model_info.txt", "w") as f:
        f.write(f"Model: {model_path}\n")
        f.write("-" * 20 + "\n")
        f.write("Inputs:\n")
        for i in session.get_inputs():
            f.write(f"  Name: {i.name}\n")
            f.write(f"  Shape: {i.shape}\n")
            f.write(f"  Type: {i.type}\n")
            f.write("\n")

        f.write("Outputs:\n")
        for o in session.get_outputs():
            f.write(f"  Name: {o.name}\n")
            f.write(f"  Shape: {o.shape}\n")
            f.write(f"  Type: {o.type}\n")
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the ONNX model signature")
    parser.add_argument("--model", required=True, help="Path to the ONNX model")
    args = parser.parse_args()

    inspect_model(args.model)
