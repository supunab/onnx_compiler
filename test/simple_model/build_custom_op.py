from __future__ import annotations

# to be able to load the scripts
import sys
import os

sys.path.insert(1, os.path.abspath("./../../"))

import onnx
from converter import compile
from custom_op_generator import generate, convert_graph

if __name__ == "__main__":
    onnx_model = "simple.onnx"
    converted_model_path = "simple_converted.onnx"
    compile_folder = "./tmp"
    model_name = "test_model"

    # TODO: here we assume a fused graph! (e.g., graph containing LayerNorm, not the Add, Reduce, etc. atomic ops for LayerNorm)
    model = onnx.load_model(onnx_model)
    context = compile(model, compile_folder, model_name)
    generate(context, os.path.join(compile_folder, model_name))
    convert_graph(model.graph, context, converted_model_path)