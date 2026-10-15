import onnx
import os

model_path = "checkpoints/ort_cache/unetexportwrapper_nvidia_geforce_rtx_4090_12_4.onnx"
if not os.path.exists(model_path):
    print(f"Error: {model_path} does not exist.")
    exit(1)

model = onnx.load(model_path)
print("Model Inputs:")
for input in model.graph.input:
    print(input.name)
