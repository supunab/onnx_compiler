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
from utils import clean_name, map_type
import numpy as np
from onnx import helper


def extract_shape(onnx_value: onnx.ValueInfoProto) -> list[int]:
    # TODO: need anything special for dynamic shapes here?
    onnx_shape = onnx_value.type.tensor_type.shape
    shape = []
    for dim in onnx_shape.dim:
        shape.append(dim.dim_value)
    return shape
    

def extract_type(onnx_value: onnx.ValueInfoProto) -> str:
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


## a wrapper for onnx graphs that contain an indexed graphical representation
class OnnxModel:
    def __init__(self, model):
        self.model = model
        self.graph = model.graph
        consumers, outputs, producers = _find_consumers(model.graph)
        self.consumers = consumers
        self.outputs = outputs # not to be confused with the actual outputs of the full model (instead, it is output of each node)
        self.producers = producers
        self.removed_nodes = set()

        self.name2init = {}
        for init in model.graph.initializer:
            self.name2init[init.name] = init
    
    def is_init(self, name: str) -> bool:
        return name in self.name2init


def _find_consumers(graph: onnx.GraphProto) -> dict:
    i = 0 # to uniquely name nodes without names
    consumers = {} # consumers for a given node's given input
    outputs = {} # outputs produced by a given node
    producers = {} # producer of a particular tensor (name)
    # populate producers with inputs and inits
    for init in graph.initializer:
        if init.name == "":
            init.name = f"node_{i}"
            i += 1
        producers[init.name] = init
        consumers[init.name] = {init.name : []}
    for input in graph.input:
        if input.name == "":
            input.name = f"node_{i}"
            i += 1
        producers[input.name] = input
        consumers[input.name] = {input.name : []}

    for node in graph.node:
        if node.name == "":
            node.name = f"node_{i}"
            i +=1 
        consumers[node.name] = {}
        output_names = list(node.output)
        outputs[node.name] = output_names
        for output in output_names:
            producers[output] = node
            consumers[node.name][output] = []
        
        input_names = list(node.input)
        for input in input_names:
            producer = producers[input]
            consumers[producer.name][input].append(node)
    return consumers, outputs, producers
            

def next_trivially_fusable_node(curr_node: onnx.NodeProto, model: OnnxModel):
    # this is only for cases where there's a single output, that's connected to single consumer
    # check if only one output
    if len(model.outputs[curr_node.name]) == 1:
        output = model.outputs[curr_node.name][0]
        # check if only one consumer
        if len(model.consumers[curr_node.name][output]) == 1:
            consumer = model.consumers[curr_node.name][output][0]
            # check if only one actual input (i.e., without constants)
            found_one_var_input = False
            for input in consumer.input:
                if not model.is_init(input):
                    if found_one_var_input:
                        return None
                    found_one_var_input = True
            # only one variable input, this is a fusable node!
            return consumer
    return None


"""
(in-place) graph level optimizations. Mostly for changing nodes (e.g., Matmul --> GEMM) and fusing (GEMM + fast_gelu) so that
           corresponding optimized backend op can be directly used
            (TODO: kind of hard coded for BERT ops right now)
"""
def optimize_graph(model: onnx.ModelProto) -> None:
    m = OnnxModel(model)
    to_remove = []
    graph = model.graph
    for node in graph.node:
        if node in to_remove:
            continue

        if node.op_type == "MatMul":
            next_node = next_trivially_fusable_node(node, m)
            if next_node != None and next_node.op_type == "FastGelu":
                # can be fused to AIT's gemm_rcr_gelu
                input_a_name = node.input[0]
                input_b_name = node.input[1]

                # check if weights is a constant
                if m.is_init(input_b_name):
                    b_init = m.name2init[input_b_name]

                    # a_shape = extract_shape(a_input) TODO: need to figure out a way to find the shape?
                    b_shape = list(b_init.dims)

                    # check if row-major (I think it must be for Matmul)
                    # assert a_shape[-1]==b_shape[-2]

                    # transpose to column-major so that we can use gemm_rcr_fast_gelu
                    d1 = b_init.dims.pop()
                    d2 = b_init.dims.pop()
                    b_init.dims.append(d1)
                    b_init.dims.append(d2)
                    # change raw data
                    data = np.frombuffer(b_init.raw_data, dtype=np.float16).reshape(b_shape) # TODO: dtype hardcoded
                    data = data.transpose()
                    b_init.raw_data = data.tobytes()

                    # remove old nodes and create new "gemm_rcr_fast_gelu"
                    to_remove.append(node)
                    to_remove.append(next_node)

                    # create node
                    new_node = helper.make_node("gemm_rcr_fast_gelu", inputs=[input_a_name, input_b_name, next_node.input[1]], outputs=[next_node.output[0]])
                    graph.node.append(new_node)
            # TODO: perhaps we can do other types of fusions here as well (e.g., for the ones that doesn't get fused automatically from AIT)
    
    for node in to_remove:
        graph.node.remove(node)
        print(f"removing {node.name}")

def compile(model: onnx.ModelProto, output_dir: str = "./tmp", model_name: str = "test_model", not_compile: bool = False):
    graph = model.graph
    context = ConverterContext(graph)
    optimize_graph(model)

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
    
    if not not_compile:
        # traverse the graph (graph is already topologically sorted)
        for node in nodes:
            process_node(node, context)
            
        # tracing is done, compile the model
        output = context.get_final_output()
        target = detect_target()
        compile_model(output, target, output_dir, model_name)
    else:
        # outputs will not be added to context since we're skipping process_node
        # add them explicitly
        for output in outputs:
            tensor = create_tensor_from_onnx_value(output)
            context.add_tensor(tensor)
    return context

    