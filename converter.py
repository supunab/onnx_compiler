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
from utils import clean_name, map_type, to_attribute_dict
import numpy as np
from onnx import helper
import logging
from onnxruntime.transformers.onnx_model import OnnxModel


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
class ModelWraper:
    def __init__(self, model):
        self.model = model
        self.graph = model.graph
        consumers, outputs, producers = ModelWraper._find_consumers(model.graph)
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
        # note - cannot assume topological ordering
        i = 0 # to uniquely name nodes without names
        consumers = {} # consumers for a given node's given output
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
            # if node.name == "BiasGelu_Approximation_0":
            #     print(output_names)
            for output in output_names:
                producers[output] = node
                consumers[node.name][output] = []
            
            input_names = list(node.input)
            for input in input_names:
                producer = producers[input]
                consumers[producer.name][input].append(node)
        return consumers, outputs, producers
            

def next_trivially_fusable_node(curr_node: onnx.NodeProto, model: ModelWraper):
    # this is only for cases where there's a single output, that's connected to single consumer
    # check if only one output
    if len(curr_node.output) == 1:
        output = curr_node.output[0]
        # print(f"curr_node: {curr_node.name}")
        # check if only one consumer
        if len(model.consumers[curr_node.name][output]) == 1:
            consumer = model.consumers[curr_node.name][output][0]
            # print(f"consumer: {consumer.name}")
            # check if only one actual input (i.e., without constants)
            found_one_var_input = False
            for input in consumer.input:
                # print(f"input: {input}; is_init: {model.is_init(input)}")
                if not model.is_init(input):
                    if found_one_var_input:
                        return None
                    found_one_var_input = True
            # only one variable input, this is a fusable node!
            return consumer
    return None


def convert_row_major_constant_to_col_major(constant: onnx.TensorProto) -> None:
    shape = list(constant.dims)
    d1 = constant.dims.pop()
    d2 = constant.dims.pop()
    constant.dims.append(d1)
    constant.dims.append(d2)
    # change raw data
    data = np.frombuffer(constant.raw_data, dtype=np.float16).reshape(shape) # TODO: dtype hardcoded
    data = data.transpose()
    constant.raw_data = data.tobytes()

"""
(in-place) graph level optimizations. Mostly for changing nodes (e.g., Matmul --> GEMM) and fusing (GEMM + fast_gelu) so that
           corresponding optimized backend op can be directly used
            (TODO: kind of hard coded for BERT ops right now)
"""
def optimize_graph(model: onnx.ModelProto) -> None:
    changed = True
    iter = 0
    while changed:
        iter += 1
        logging.info(f"Running graph optimize iteration {iter}")
        m = ModelWraper(model)
        graph = model.graph
        changed = False
        to_remove = []
        for node in graph.node:
            if node in to_remove:
                continue

            # TODO: remove casts at the boundary 

            # MatMul -> bias -> fast gelu fusion
            # Matmul -> add -> add fusion
            if node.op_type == "MatMul":
                next_node = next_trivially_fusable_node(node, m)
                # ORT `FastGelu` node has contains the bias for the prev linear as well
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
                        convert_row_major_constant_to_col_major(b_init)

                        # remove old nodes and create new "gemm_rcr_fast_gelu"
                        to_remove.append(node)
                        to_remove.append(next_node)

                        # create node
                        new_node = helper.make_node("gemm_rcr_fast_gelu", inputs=[input_a_name, input_b_name, next_node.input[1]], outputs=[next_node.output[0]])
                        graph.node.append(new_node)
                        changed = True
                        continue
                
                # if B is a constant (so that we can make it col-major) and next op is add, we can
                # try gemm_rcr_add_add fusion
                if next_node != None and m.is_init(node.input[1]) and next_node.op_type == "Add":
                    # see if the consumer is also an add
                    add1_output = next_node.output[0]
                    add1_consumers = m.consumers[next_node.name][add1_output]

                    if len(add1_consumers) == 1 and add1_consumers[0].op_type == "Add":
                        next_next_node = add1_consumers[0]
                        # can do gemm_rcr_add_add fusion
                        # convert Matmul's B to col-major (to use gemm_rcr)
                        matmul_a_name = node.input[0]
                        matmul_b_name = node.input[1]

                        matmul_b = m.name2init[matmul_b_name]
                        convert_row_major_constant_to_col_major(matmul_b)

                        add1_other_input = next_node.input[0] if next_node.input[0] != node.output[0] else node.input[1]
                        add2_other_input = next_next_node.input[0] if next_next_node.input[0] != next_node.output[0] else next_next_node.input[1]

                        final_output_name = next_next_node.output[0]
                        # create new node
                        gemm_rcr_add_add_node = helper.make_node(
                            op_type="gemm_rcr_add_add_node",
                            inputs=[matmul_a_name, matmul_b_name, add1_other_input, add2_other_input],
                            outputs=[final_output_name],
                            name=node.name + "_add_add_fused"
                        )
                        graph.node.append(gemm_rcr_add_add_node)
                        to_remove.extend([node, next_node, next_next_node])
                        changed = True


            elif node.op_type == "SkipLayerNormalization":
                # unpack SkipLayerNorm for better fusion (in AIT)
                res_input_name = node.input[0]
                matmul_output_name = node.input[1]
                ln_weight_name = node.input[2]
                ln_bias_name = node.input[3]
                prev_bias_weight_name = node.input[4]

                output_name = node.output[0]

                bias_add_node_output_name = matmul_output_name + "_bias_added"
                # create add node (for bias)
                bias_add_node = helper.make_node(
                    op_type="Add",
                    inputs=[matmul_output_name, prev_bias_weight_name],
                    outputs=[bias_add_node_output_name],
                    name=matmul_output_name + "_add_bias"
                )

                # create add node (bias_output + residual_input)
                residual_add_output_name = matmul_output_name + "_bias_residual_added"
                residual_add_node = helper.make_node(
                    op_type="Add",
                    inputs=[bias_add_node_output_name, res_input_name],
                    outputs=[residual_add_output_name],
                    name=matmul_output_name + "_added_bias_add_residual"
                )

                # create layernorm node
                ln_node = helper.make_node(
                    op_type="LayerNormalization",
                    inputs=[residual_add_output_name, ln_weight_name, ln_bias_name],
                    outputs=[output_name],
                    name=output_name + "_ln_node"
                )
                # find attributes and add them to ln_node
                for attr in node.attribute:
                    if attr in ["axis", "epsilon", "stash_type"]:
                        ln_node.attribute.append(attr)

                # add newly created nodes
                graph.node.extend([bias_add_node, residual_add_node, ln_node])
                to_remove.append(node)
                changed = True

            ## TODO: add case for matmul with b as init to gemm_rcr_

                # TODO: perhaps we can do other types of fusions here as well (e.g., for the ones that doesn't get fused automatically from AIT)
        
        for node in to_remove:
            graph.node.remove(node)
            logging.debug(f"removing {node.name}")

        # need to topologicaly sort since newly added nodes are just prepended
        if changed:
            OnnxModel.graph_topological_sort(graph)

    logging.info("Compelted graph optimize pass")

def compile(model: onnx.ModelProto, output_dir: str = "./tmp", model_name: str = "test_model", not_compile: bool = False, attributes = {}) -> ConverterContext:
    """
    Compile the model to AIT model ops. The returned converter context contains all the input, intermediate, and output AIT Tensors. This 
    converter context then can be used in the rest of the code generation process (i.e., generating the custom op).
    Args
    model: the onnx model
    output_dir: where the AIT generated source code should be stored
    model_name: name of the model (used as the folder name for the generated sources directory)
    not_compile: only populate the ConverterContext (with inputs, inits, and ouputs; without intermediates). 
                Used when AIT generated sources are not needed (e.g., they are generated separately)
    attributes: attributes of the model that might not be available on the onnx graph (e.g., batch size, hidden size, etc.)
    """
    graph = model.graph
    context = ConverterContext(graph, attributes)
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

    