import onnxruntime
import argparse
import sys

def inspect_model(model_path):
    try:
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return

    with open("model_info.txt", "w") as f:
        f.write(f"Model: {model_path}\n")
        f.write("-" * 20 + "\n")
        f.write("Inputs:\n")
        for i in session.get_inputs():
            f.write(f"  Nome: {i.name}\n")
            f.write(f"  Shape: {i.shape}\n")
            f.write(f"  Tipo: {i.type}\n")
            f.write("\n")

        f.write("Outputs:\n")
        for o in session.get_outputs():
            f.write(f"  Nome: {o.name}\n")
            f.write(f"  Shape: {o.shape}\n")
            f.write(f"  Tipo: {o.type}\n")
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspeção da assinatura do modelo ONNX")
    parser.add_argument("--model", required=True, help="Caminho para o modelo ONNX")
    args = parser.parse_args()

    inspect_model(args.model)
