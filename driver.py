import onnx
from converter import convert
from custom_op_generator import generate, convert_graph

if __name__ == "__main__":
    onnx_model = "simple.onnx"
    # TODO: here we assume a fused graph! (e.g., graph containing LayerNorm, not the Add, Reduce, etc. atomic ops for LayerNorm)
    model = onnx.load_model(onnx_model)
    context = convert(model)
    generate(context)
    convert_graph(model.graph, context)

