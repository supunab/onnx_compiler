import onnx
from converter import convert

if __name__ == "__main__":
    onnx_model = "simple.onnx"
    # TODO: here we assume a fused graph! (e.g., graph containing LayerNorm, not the Add, Reduce, etc. atomic ops for LayerNorm)
    model = onnx.load_model(onnx_model)
    convert(model)

