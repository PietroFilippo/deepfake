import onnx
from onnxconverter_common import float16

input_model_path = "models/inswapper_128.onnx"
output_model_path = "models/inswapper_128_fp16.onnx"

print(f"Carregando modelo de {input_model_path}.")
try:
    model = onnx.load(input_model_path)
    print("Modelo carregado com sucesso.")

    print("Convertendo para FP16.")
    # keep_io_types=True garante que entradas/saídas permaneçam float32 se necessário para compatibilidade
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    
    print(f"Salvando modelo FP16 em {output_model_path}.")
    onnx.save(model_fp16, output_model_path)
    print("Conversão completa.")

except Exception as e:
    print(f"Erro durante a conversão: {e}")
