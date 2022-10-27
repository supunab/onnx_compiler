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


def map_onnx_dtype_to_numpy(onnx_dtype: int):
    if onnx_dtype == 10:
        dtype = np.float16
    else:
        raise NotImplementedError("only float16 is handled for now")


# this is special handling logic for "[*]ff.net.0.proj.weight" and "[*]ff.net.0.proj.weight"
# essentially a translation of https://github.com/facebookincubator/AITemplate/blob/d0ee90156f218f5d39007b1fabdb299cebbeac3b/examples/05_stable_diffusion/compile.py#L51
def special_handling(graph: onnx.GraphProto, dim: int) -> dict:
    new_inits = {}
    for init in graph.initializer:
        shape = list(init.dims)
        if shape == 4:
            # get the raw_data as a numpy array
            dtype = map_onnx_dtype_to_numpy(init.data_type)
            np_array = np.frombuffer(init.raw_data, dtype=dtype).reshape(shape)
            permuted_array = np_array.transpose(0, 2, 3, 1)
            # TODO: double check whether the default memory layout is correct?
            bytes = permuted_array.tobytes()
            init.raw_data = bytes

        if init.name.endswith("ff.net.0.proj.weight") or init.name.endswith("ff.net.0.proj.weight"):
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
    context.tensors["arange"] = Tensor(shape=arange.shape, name="arange", dtype=np.float16)
    return new_inits


if __name__ == "__main__":
    # load the onnx model
    model_path = "/work/models/sd_unet.onnx"
    generated_header_location = "./"
    onnx_model = onnx.load_model(model_path)
    graph = onnx_model.graph
    context = compile(onnx_model, not_compile = True) # not actually compiling here (we use the source generated from AIT directly instead)

    # need special handling as specified in https://github.com/facebookincubator/AITemplate/blob/d0ee90156f218f5d39007b1fabdb299cebbeac3b/examples/05_stable_diffusion/compile.py#L51
    ## TODO: what is dims?
    dim = 320 # this is the default value used for unet here: https://github.com/facebookincubator/AITemplate/blob/d0ee90156f218f5d39007b1fabdb299cebbeac3b/examples/05_stable_diffusion/compile.py#L182
    special_inits = special_handling(onnx_model.graph, dim)
    generate(context, generated_header_location)
    convert_graph(graph, context, "/work/models/sd_unet_converted.onnx", special_inits)
