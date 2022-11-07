"""
Create ONNX graph that contains a single custom op, all inputs, and all the weights as initializers
"""
from __future__ import annotations

import sys
import os
sys.path.insert(1, os.path.abspath("./../../"))

from common import *
import torch
import torch.nn as nn
import numpy as np
import onnx
from onnx import helper
from utils import clean_name, map_np_type_to_onnx
from converter import compile
from custom_op_generator import generate
import logging
from tqdm import tqdm
import subprocess
import click
import math


def create_initializer_special(name: str, param: nn.parameter.Parameter) -> list[onnx.TensorProto]:
    data = param.detach().cpu().numpy()
    shape = data.shape
    # special handling (https://github.com/facebookincubator/AITemplate/blob/d0ee90156f218f5d39007b1fabdb299cebbeac3b/examples/05_stable_diffusion/compile.py#L51)
    if shape == 4:
        # need to permute since AIT kernels expect permuted shape
        data = data.transpose(0, 2, 3, 1)

    if name.endswith("ff.net.0.proj.weight") or name.endswith("ff.net.0.proj.bias"):
        # need to split the original data into two chunks
        chunk_point = math.ceil(shape[0] / 2)
        data_proj = data.take(range(chunk_point), axis=0)
        data_gate = data.take(range(chunk_point, shape[0]), axis=0)
        name_proj = name
        name_gate = name.replace("proj", "gate")

        init_proj = helper.make_tensor(clean_name(name_proj), map_np_type_to_onnx(data_proj.dtype), data_proj.shape, data_proj.tobytes(), raw=True)
        init_gate = helper.make_tensor(clean_name(name_gate), map_np_type_to_onnx(data_gate.dtype), data_gate.shape, data_gate.tobytes(), raw=True)

        # these two are now essentially two separate params
        return [init_proj, init_gate]

    # create the node
    return [helper.make_tensor(clean_name(name), map_np_type_to_onnx(data.dtype), data.shape, data.tobytes(), raw=True)]

def log_memory_usage():
    result = subprocess.check_output(['bash','-c', 'free -h'])
    free_memory = str(result).split('\\n')[1].split()[3]
    logging.info(f"Free memory {free_memory}")

def create_sd_onnx_graph(model_path: str):
    # load hf sd model
    logging.info("Obtaining model from hugging face")
    pt_model = get_pt_sd_unet_from_hf()
    logging.info("Model obtained")
    log_memory_usage()

    # create initializers from each weight
    logging.info("Creating initializer nodes")
    inits = []
    for name, param in tqdm(pt_model.named_parameters()):
        inits += create_initializer_special(name, param)

    del pt_model
    import gc
    gc.collect()
    # need a arange input as well
    arange = np.arange(start = 0, stop=dim // 2, dtype=np.float16)
    arange_init = helper.make_tensor("arange", onnx.TensorProto.FLOAT16, arange.shape, arange)
    inits.append(arange_init)
    logging.info("Creation of initializers completed")
    log_memory_usage()

    # create input and ouputs
    # has three inputs: latent_model_input[batch_size, channels, hh, ww]float16, timesteps[batch_size]float16 (all 1's?), text_embeddings[batch_size, prompt_length, emb_size]float16
    # has one output: [batch_size, input_channels, hh, ww]float16
    # requires the inputs to be permuted!
    logging.info("Creating input/output nodes")
    onnx_float16 = map_np_type_to_onnx(np.float16)
    input0 = helper.make_tensor_value_info("input0", onnx_float16, [batch_size, hh, ww, input_channels]) # latent_model_input_permuted
    input1 = helper.make_tensor_value_info("input1", onnx_float16, [batch_size]) # timesteps
    input2 = helper.make_tensor_value_info("inpupt2", onnx_float16, [batch_size, prompt_length, embedding_size]) # text_embeddings
    inputs = [input0, input1, input2]

    output = helper.make_tensor_value_info("output", onnx_float16, [batch_size, hh, ww, input_channels]) # output image
    
    logging.info("Completing the graph")
    # order of inputs + inits is required because the generated code retrieves op inputs in that order
    input_and_init_names = list(map(lambda n: n.name, [input0, input1, input2] + inits))
    output_names = [output.name]
    # create the graph
    node = helper.make_node(
        op_type = "AITModelOp",
        inputs = input_and_init_names,
        outputs = output_names,
        name = "generated_custom_op",
        domain = "ait.customop"
    )

    graph = helper.make_graph(
        nodes = [node, ],
        name = "test-model",
        inputs = inputs,
        outputs = [output],
        initializer = inits
    )

    log_memory_usage()

    logging.info("Creating the model")
    model = helper.make_model(graph, producer_name="ait-customop-generator")
    logging.info("Saving the model")

    # from onnx.external_data_helper import convert_model_to_external_data, write_external_data_tensors
    # convert_model_to_external_data(model, all_tensors_to_one_file=False, size_threshold=0, convert_attribute=True)
    # onnx.save_model(model, model_path, save_as_external_data=True)
    
    onnx.save_model(model, model_path)

def generate_custom_op(model: onnx.ModelProto, folder: str) -> None:
    context = compile(model, not_compile=True)

    # TODO: below is a hack, we have to manually look at the shapes and do the matching --> automate this!
    # input orders to ort custom op and AIT doesn't match for this model, hence, need inputs_order
    # look at param[0], param[1], and param[2] in the generated code. This is the order in which we should pass ait inputs
    inputs_order = [1, 0, 2]
    generate(context, folder=folder, run_make=False, inputs_order=inputs_order)


def make_so(generated_cu: str, generated_h: str, ait_sources: str):
    ret = os.system(f"cd {ait_sources} && make")
    if ret != 0:
        raise Exception("Error: returned non-zero when trying to compile (i.e., make) the generated sources")


@click.command()
@click.option("--use_prev_model", is_flag=True)
@click.option("--model_path", default="./sd_unet_converted.onnx", help="Directory where the model with custom op should be saved")
@click.option("--ait_sources", default="./ait_generated_sources/", help="Directory containing the generated sources from AIT. \
                                                                    This is where the sources for ort custom op will be created")
def _run(use_prev_model, model_path: str, ait_sources: str):
    logging.Logger.setLevel(logging.getLogger(), logging.INFO)
    if not use_prev_model:
        create_sd_onnx_graph(model_path)

    logging.info("Loading the onnx model with custom op")
    model = onnx.load_model(model_path)
    logging.info("Generating custom op wrapper header and cu")
    generate_custom_op(model, folder=ait_sources)

    logging.info("Compiling and making custom op shared object")
    generated_cu = os.path.join(ait_sources, "ort_ait_custom_op_library.cu")
    generated_h = os.path.join(ait_sources, "ort_ait_custom_op_library.h")
    make_so(generated_cu, generated_h, ait_sources)

if __name__ == "__main__":
    _run()