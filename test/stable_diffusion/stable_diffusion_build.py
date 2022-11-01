"""
Trying to get stable diffusion UNet working from onnx graph (with weights) by using the
generated AIT code wrapped as an ORT custom op
"""

from __future__ import annotations

# to be able to load the scripts
import sys
import os

sys.path.insert(1, os.path.abspath("./../../"))

from converter import compile
from utils import clean_name
import numpy as np
import onnx
from onnx import helper, TensorProto
from aitemplate.frontend import Tensor
from custom_op_generator import convert_graph
from custom_op_generator import generate
from constants import *
from utils import map_onnx_dtype_to_numpy


# this is special handling logic for "[*]ff.net.0.proj.weight" and "[*]ff.net.0.proj.weight"
# essentially a translation of https://github.com/facebookincubator/AITemplate/blob/d0ee90156f218f5d39007b1fabdb299cebbeac3b/examples/05_stable_diffusion/compile.py#L51
def special_handling(graph: onnx.GraphProto, dim: int) -> dict:
    new_inits = {}
    for init in graph.initializer:
        shape = list(init.dims)
        if shape == 4: 
            # this is because generated AIT kernel expects (batch, h, w, channel)
            # whereas pytorch tensor is (batch, channel, h, w) 

            # get the raw_data as a numpy array
            dtype = map_onnx_dtype_to_numpy(init.data_type)
            np_array = np.frombuffer(init.raw_data, dtype=dtype).reshape(shape)
            permuted_array = np_array.transpose(0, 2, 3, 1)
            # TODO: double check whether the default memory layout is correct?
            bytes = permuted_array.tobytes()
            init.raw_data = bytes

        if init.name.endswith("ff.net.0.proj.weight") or init.name.endswith("ff.net.0.proj.bias"):
            # needs special handling
            ait_name = clean_name(init.name)
            assert (ait_name in context.initializers)
            # remove the original tensor from context
            del context.tensors[ait_name]
            context.initializers.remove(ait_name)
            dtype = map_onnx_dtype_to_numpy(init.data_type)
            np_array = np.frombuffer(init.raw_data, dtype=dtype).reshape(shape)
            # chunk them into two arrays
            chunk_point = shape[0] // 2
            array_proj = np_array.take(range(chunk_point), axis=0)
            array_gate = np_array.take(range(chunk_point, shape[0]), axis=0)
            # add this to initializers
            array_proj_name = ait_name
            array_gate_name = ait_name.replace("proj", "gate")
            context.initializers.extend([array_proj_name, array_gate_name])
            context.tensors[array_proj_name] = Tensor(shape=array_proj.shape, name=array_proj_name, dtype=dtype)
            context.tensors[array_gate_name] = Tensor(shape=array_gate.shape, name=array_gate_name, dtype=dtype)
            # create TensorProto
            new_inits[array_proj_name] = helper.make_tensor(array_proj_name, init.data_type, array_proj.shape, array_proj)
            new_inits[array_gate_name] = helper.make_tensor(array_gate_name, init.data_type, array_gate.shape, array_gate)
    
    # add the "arange" tensor as well
    arange = np.arange(start = 0, stop=dim // 2, dtype=np.float16)
    new_inits["arange"] = helper.make_tensor("arange", TensorProto.FLOAT16, arange.shape, arange)
    # update the context
    context.initializers.append("arange")
    context.tensors["arange"] = Tensor(shape=arange.shape, name="arange", dtype="float16")
    return new_inits


if __name__ == "__main__":
    # load the onnx model
    model_path = "/work/models/sd_unet.onnx"
    generated_header_location = "./"
    onnx_model = onnx.load_model(model_path)
    graph = onnx_model.graph

    # AIT stable diffusion implementation assumes input to be in [batch_size, hh, ww, num_channels] whereas PyTorch expects [batch_size, num_channels, hh, ww]
    # hence, update the inputs of the graph to reflect this permute
    # (we have to make sure that we pass the permuted input to the compiled custom op)
    graph.input[0].type.tensor_type.shape.dim[0].dim_value = batch_size
    graph.input[0].type.tensor_type.shape.dim[1].dim_value = hh
    graph.input[0].type.tensor_type.shape.dim[2].dim_value = ww
    graph.input[0].type.tensor_type.shape.dim[3].dim_value = input_channels


    context = compile(onnx_model, not_compile = True) # not actually compiling here (we use the source generated from AIT directly instead)

    # need special handling as specified in https://github.com/facebookincubator/AITemplate/blob/d0ee90156f218f5d39007b1fabdb299cebbeac3b/examples/05_stable_diffusion/compile.py#L51
    ## TODO: what is dims?
    dim = 320 # this is the default value used for unet here: https://github.com/facebookincubator/AITemplate/blob/d0ee90156f218f5d39007b1fabdb299cebbeac3b/examples/05_stable_diffusion/compile.py#L182
    special_inits = special_handling(onnx_model.graph, dim)

    # explicitly specify output shape
    output_name = context.get_all_outputs()[0]._attrs["name"]
    output_shape = { output_name : [batch_size, input_channels, hh, ww]}

    # TODO: below is a hack, we have to manually look at the shapes and do the matching --> automate this!
    # input orders to ort custom op and AIT doesn't match for this model, hence, need inputs_order
    # look at param[0], param[1], and param[2] in the generated code. This is the order in which we should pass ait inputs
    inputs_order = [1, 0, 2]
    generate(context, generated_header_location, output_shape=output_shape, inputs_order=inputs_order)
    convert_graph(graph, context, "/work/models/sd_unet_converted.onnx", special_inits)
