"""
Converts an ONNX graph into an AITemplate graph
"""
from __future__ import annotations

import onnx
from aitemplate.frontend import Tensor
from aitemplate.compiler import compile_model
from aitemplate.testing import detect_target
from converter_context import ConverterContext, ModelWraper
from registry import process_node
from utils import clean_name, map_type, to_attribute_dict, map_np_type_to_onnx, map_onnx_dtype_to_numpy
import numpy as np
from onnx import helper, numpy_helper
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
    dtype = map_onnx_dtype_to_numpy(constant.data_type)
    data = np.frombuffer(constant.raw_data, dtype=dtype).reshape(shape)
    # only transpose the last two dims
    perm = np.arange(0, len(shape)).tolist()
    perm[-1], perm[-2] = perm[-2], perm[-1]
    data = data.transpose(perm)
    constant.raw_data = data.tobytes()

def remove_attention_mask_hack(model: onnx.ModelProto):
    for node in model.graph.node:
        if node.op_type == "EmbedLayerNormalization":
            # remove the 8th (last) input
            name = node.input[7]
            node.input.remove(name)
            # remove from graph inputs
            to_remove = None
            for input in model.graph.input:
                if name == input.name:
                    to_remove = input
                    break
            model.graph.input.remove(to_remove)
            break


def clean_name_graph(model: onnx.ModelProto):
    """
    Convert all the names (inputs, inits, outputs) to the "cleaned" name using clean_name()
    """
    graph = model.graph
    # clean inputs
    for input in graph.input:
        input.name = clean_name(input.name)
    
    # clean outputs
    for output in graph.output:
        output.name = clean_name(output.name)

    # clean inits
    for init in graph.initializer:
        init.name = clean_name(init.name)

    # clean node inputs and outputs
    for node in graph.node:
        for i in range(len(node.output)):
            node.output[i] = clean_name(node.output[i])
        for i in range(len(node.input)):
            node.input[i] = clean_name(node.input[i])
    

"""
(in-place) graph level transformations. Mostly for changing nodes (e.g., Matmul --> GEMM) and fusing (GEMM + fast_gelu) so that
           corresponding optimized backend op can be directly used
            (TODO: kind of hard coded for BERT ops right now)

            attributes: a dict of attributes like seq_len, batch_size, etc.
"""
def transform_graph(model: onnx.ModelProto, attributes: dict) -> None:
    # convert all symbolic shapes to fixed shapes (TODO: don't support any form of dynamic shape atm)
    for input in model.graph.input:
        for dim in input.type.tensor_type.shape.dim:
            if dim.dim_param != "":
                # symbolic shape, query the actual value and update
                dim.dim_value = attributes[dim.dim_param]

    # clean names for convinience
    clean_name_graph(model)

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

            # remove fp32 <-> fp16 casts at the boundary, we'll assume input and outputs are fp16
            if node.op_type == "Cast" and iter == 1:
                cast_input_name = node.input[0]
                cast_output_name = node.output[0]
                cast_to = node.attribute[0].i

                # check if input is an input to the model
                for i in graph.input:
                    if i.name == cast_input_name:
                        # it is casting an input, remove the cast
                        # first update all the consumers to use the input directly
                        consumers = m.consumers[node.name][cast_output_name]
                        for consumer in consumers:
                            # directly connect to the input (all uses)
                            consumer_new_input_list = []
                            while len(consumer.input):
                                t = consumer.input.pop()
                                if t == cast_output_name:
                                    consumer_new_input_list.append(cast_input_name)
                                else:
                                    consumer_new_input_list.append(t)
                            # add them back
                            while len(consumer_new_input_list):
                                consumer_input = consumer_new_input_list.pop()
                                consumer.input.append(consumer_input)

                            # update the input data type
                            i.type.tensor_type.elem_type = cast_to

                            # remove the cast node
                            to_remove.append(node)

                        break
                else:
                    # or, check if the output is an output of the model
                    for o in graph.output:
                        if o.name == cast_output_name:
                            # cast is directly used by an output
                            # TODO: right now, skipping if it has more uses by other intermediate nodes
                            assert len(m.consumers[node.name][cast_output_name]) == 0 # just an output, no consumers
                            o.name = cast_input_name
                            # how to find the original type
                            o.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
                            logging.warn("TODO: output cast dtype is hardcoded to be float16")
                            to_remove.append(node)
                            break
                

            # MatMul -> bias -> fast gelu fusion
            # Matmul -> add -> add fusion
            elif node.op_type == "MatMul":
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
                # try gemm_rcr_bias_add fusion
                if next_node != None and m.is_init(node.input[1]) and next_node.op_type == "Add":
                    # see if the consumer is also an add
                    add1_output = next_node.output[0]
                    add1_consumers = m.consumers[next_node.name][add1_output]

                    if len(add1_consumers) == 1 and add1_consumers[0].op_type == "Add":
                        next_next_node = add1_consumers[0]
                        # can do gemm_rcr_bias_add fusion
                        # convert Matmul's B to col-major (to use gemm_rcr)
                        matmul_a_name = node.input[0]
                        matmul_b_name = node.input[1]

                        matmul_b = m.name2init[matmul_b_name]
                        convert_row_major_constant_to_col_major(matmul_b)

                        add1_other_input = next_node.input[0] if next_node.input[0] != node.output[0] else next_node.input[1]
                        add2_other_input = next_next_node.input[0] if next_next_node.input[0] != next_node.output[0] else next_next_node.input[1]

                        final_output_name = next_next_node.output[0]
                        # create new node
                        gemm_rcr_bias_add_node = helper.make_node(
                            op_type="gemm_rcr_bias_add",
                            inputs=[matmul_a_name, matmul_b_name, add1_other_input, add2_other_input],
                            outputs=[final_output_name],
                            name=node.name + "_add_add_fused"
                        )
                        graph.node.append(gemm_rcr_bias_add_node)
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
                    if attr.name in ["axis", "epsilon", "stash_type"]:
                        ln_node.attribute.append(attr)

                # add newly created nodes
                graph.node.extend([bias_add_node, residual_add_node, ln_node])
                to_remove.append(node)
                changed = True

            # TODO: len checking might not be the most accurate thing to do here
            elif node.op_type == "EmbedLayerNormalization" and len(node.input) == 8:
                # this is not really an optimization, but if this doesn't have position_ids, add a default value
                seq_len = attributes["seq_len"]
                batch_size = attributes["batch_size"]
                dtype = np.int32
                data = np.arange(0, seq_len, dtype=dtype).reshape(1, seq_len).repeat(batch_size, axis=0)
                init_node_name = "__default_pos_ids"
                # check if this already exists (e.g., used for a different node)
                found = False
                for init in graph.initializer:
                    if init.name == init_node_name:
                        found = True
                        break
                if not found:
                    init_node = helper.make_tensor(
                        name = init_node_name,
                        data_type=map_np_type_to_onnx(dtype),
                        dims=[batch_size, seq_len],
                        vals=data.tobytes(), raw=True)
                    # add this to inits
                    graph.initializer.append(init_node)
                
                # augment the EmbedLayerNormalization node to have this as input
                node.input.append(init_node_name)

            # convert Constant into initializers
            elif node.op_type == "Constant":
                assert node.attribute[0].name == "value" # any other cases are not properly handled here
                tensor_proto = node.attribute[0].t
                name = node.output[0]
                tensor_proto.name = name
                graph.initializer.append(tensor_proto)
                to_remove.append(node)

            # need to convert attention qkv_weight to column-major
            elif node.op_type == "Attention":
                qkv_weight_name = node.input[1]
                qkv_bias_name = node.input[2]
                qkv_weight_init = m.name2init[qkv_weight_name]
                qkv_bias_init = m.name2init[qkv_bias_name]

                weight_shape = list(qkv_weight_init.dims)
                bias_shape = list(qkv_bias_init.dims)
                
                # check if weight is row-major
                if weight_shape[-1] == bias_shape[-1]:
                    # convert to column-major
                    # TODO: for some reason values of the qkv_weight is not in raw_data, but as int32_data array
                    #       hence, convert them to bytes
                    if qkv_weight_init.raw_data == b'':
                        data_bytes = numpy_helper.to_array(qkv_weight_init).tobytes()
                        qkv_weight_init.raw_data = data_bytes
                        # remove old int32_data
                        del qkv_weight_init.int32_data[:len(qkv_weight_init.int32_data)]

                    convert_row_major_constant_to_col_major(qkv_weight_init)

            ## TODO: add case for matmul with b as init to gemm_rcr_
            ## TODO: perhaps we can do other types of fusions here as well (e.g., for the ones that doesn't get fused automatically from AIT)
        
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
    transform_graph(model, attributes)
    # TODO: hack, do this properly
    remove_attention_mask_hack(model)
    modelw = ModelWraper(model)
    context = ConverterContext(graph, modelw, attributes)

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

    