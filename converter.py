"""
Converts an ONNX graph into an AITemplate graph
"""
from __future__ import annotations

import onnx
from aitemplate.frontend import Tensor
from aitemplate.compiler import compile_model
from aitemplate.testing import detect_target
from converter_context import ConverterContext
from registry import process_node
from utils import clean_name


def extract_shape(onnx_value: onnx.ValueInfoProto) -> list[int]:
    # TODO: need anything special for dynamic shapes here?
    onnx_shape = onnx_value.type.tensor_type.shape
    shape = []
    for dim in onnx_shape.dim:
        shape.append(dim.dim_value)
    return shape

def map_type(elem_type: int) -> str:
    # TODO: need to map elem_type to correct type!
    return "float16"


def extract_type(onnx_value: onnx.ValueInfoProto) -> str:
    # TODO: hardcoded to return float16!
    elem_type = onnx_value.type.tensor_type.elem_type
    return map_type(elem_type)


def create_tensor_from_onnx_value(onnx_value: onnx.ValueInfoProto) -> Tensor:
    name = clean_name(onnx_value.name)
    shape = extract_shape(onnx_value)
    dtype = extract_type(onnx_value)
    return Tensor(shape=shape, name=name, dtype=dtype)
    

def create_tensor_from_onnx_init(onnx_init: onnx.TensorProto) -> Tensor:
    name = clean_name(onnx_init.name)
    shape = list(onnx_init.dims)
    dtype = map_type(onnx_init.data_type)
    return Tensor(shape=shape, name=name, dtype=dtype)


def convert(model: onnx.ModelProto):
    graph = model.graph
    context = ConverterContext(graph)

    # some notes
    nodes = graph.node
    inputs = graph.input
    outputs = graph.output
    inits = graph.initializer # constants/weigths
    # note - Will keep the inits as it is. Only thing we need to do is to set their names
    #       properly as AIT constants. The converted onnx graph that contains the custom op
    #       will still hold the correct initializer values and will be loaded correctly in
    #       ORT if the names are properly mapped 

    # setup inputs
    for input in inputs:
        tensor = create_tensor_from_onnx_value(input)
        # mark it as an input
        tensor._attrs["is_input"] = True
        context.add_tensor(tensor)

    # setup inits
    for init in inits:
        tensor = create_tensor_from_onnx_init(init)
        # constants/weights shouldn't be marked as inputs
        context.add_tensor(tensor)

    # traverse the graph (graph is already topologically sorted)
    for node in nodes:
        process_node(node, context)

    # tracing is done, compile the model
    output = context.get_final_output()
    target = detect_target()
    compile_model(output, target, "./tmp", "test_model")

    