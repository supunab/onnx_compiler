"""
Testing the correctness of EmbedLayerNormalization
"""
from __future__ import annotations
import sys
import os
import logging

sys.path.insert(1, os.path.abspath("./../../../"))
import click
import numpy as np
from utils import map_np_type_to_onnx
import onnx
import onnxruntime as ort


batch_size = 2
seq_len = 512
hidden_size = 768
embed_size = 768
vocab_size = 28996
token_types = 2
epsilon = 9.999999960041972e-13

# batch_size = 1
# seq_len = 1
# hidden_size = 8
# embed_size = 8
# vocab_size = 4
# token_types = 2

ait_build_folder = "./tmp/"
model_name = "ELN_test"
converted_model = "eln_converted.onnx"

def create_graph():
    from onnx import helper
    from onnx import TensorProto
    # setup inputs
    input_names = ["input_ids", "token_type_ids", "word_embedding", "pos_embedding", "token_type_embedding", 
                "ln_weight", "ln_bias", "attention_mask"]

    input_ids = helper.make_tensor_value_info(input_names[0], TensorProto.INT32, [batch_size, seq_len])
    token_type_ids = helper.make_tensor_value_info(input_names[1], TensorProto.INT32, [batch_size, seq_len])
    attention_mask = helper.make_tensor_value_info(input_names[7], TensorProto.INT32, [batch_size, seq_len])
    inputs = [input_ids, token_type_ids, attention_mask]
    

    # setup inits (constants)
    dtype = np.float16
    onnx_dtype = map_np_type_to_onnx(dtype)

    def _make_init(name: str, shape: list):
        np_data = np.random.rand(*shape).astype(dtype)
        return helper.make_tensor(name, onnx_dtype, np_data.shape, np_data.tobytes(), raw=True)

    word_embedding_init = _make_init(input_names[2], [vocab_size, embed_size])
    pos_embedding_init = _make_init(input_names[3], [seq_len, embed_size])
    token_type_embedding_init = _make_init(input_names[4], [token_types, embed_size])
    ln_weight_init = _make_init(input_names[5], [hidden_size])
    ln_bias_init = _make_init(input_names[6], [hidden_size])
    inits = [word_embedding_init, pos_embedding_init, token_type_embedding_init, ln_weight_init, ln_bias_init]
    
    output_names = ["elm_output", "mask_index"]
    node = helper.make_node(
        op_type="EmbedLayerNormalization",
        inputs = input_names,
        outputs = output_names,
        name="elm",
        domain="com.microsoft",
        epsilon=epsilon
    )

    # outputs
    elm_output = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT16, [batch_size, seq_len, hidden_size])
    mask_index_output = helper.make_tensor_value_info(output_names[1], TensorProto.INT32, [batch_size])
    outputs = [elm_output, mask_index_output]

    graph = helper.make_graph([node], "elm-graph", inputs, outputs, inits)
    model = helper.make_model(graph)
    onnx.save_model(model, "elm_model.onnx")
    return model


def build_ait_custom_op():
    from converter import transform_graph, compile
    from custom_op_generator import generate, convert_graph
    
    model = create_graph()

    # remove mask_index output since it's not an output in AIT
    model.graph.output.pop()
    model.graph.node[0].output.pop()

    attributes={
        "batch_size": batch_size,
        "hidden_size": hidden_size, 
        "seq_len": seq_len}

    context = compile(model, ait_build_folder, model_name, attributes=attributes)
    generate(context, os.path.join(ait_build_folder, model_name))
    convert_graph(model.graph, context, converted_model)


def _bind_io_input(io_binding: ort.IOBinding, name: str, data_np: np.ndarray)->None:
    data_ort = ort.OrtValue.ortvalue_from_numpy(data_np, "cuda", 0)
    io_binding.bind_input(name=name, device_type=data_ort.device_name(), device_id=0, element_type=data_np.dtype,
                            shape=data_ort.shape(), buffer_ptr=data_ort.data_ptr())

def _bind_io_output(io_binding: ort.IOBinding, name: str, ort_value: ort.OrtValue, elem_type)->None:
    io_binding.bind_output(name=name, device_type=ort_value.device_name(), device_id=0, element_type=elem_type,
                            shape=ort_value.shape(), buffer_ptr=ort_value.data_ptr())

def ort_io_binding(session: ort.InferenceSession, input_ids_np: np.ndarray, attention_mask_np: np.ndarray, token_type_np: np.ndarray, one_output: bool, no_attention_mask: bool):
    io_binding = session.io_binding()
    input_ids_name = session.get_inputs()[0].name
    assert(input_ids_name == "input_ids")
    _bind_io_input(io_binding, input_ids_name, input_ids_np)

    token_type_ids_name = session.get_inputs()[1].name
    assert(token_type_ids_name == "token_type_ids")
    _bind_io_input(io_binding, token_type_ids_name, token_type_np)

    if not no_attention_mask:
        attention_mask_name = session.get_inputs()[2].name
        assert(attention_mask_name == "attention_mask")
        _bind_io_input(io_binding, attention_mask_name, attention_mask_np)

    elm_output_name = session.get_outputs()[0].name
    assert(elm_output_name == "elm_output")
    elm_output_ort = ort.OrtValue.ortvalue_from_shape_and_type([batch_size, seq_len, hidden_size], np.float16, "cuda", 0)
    _bind_io_output(io_binding, elm_output_name, elm_output_ort, np.float16)

    if not one_output:
        mask_index_name = session.get_outputs()[1].name
        mask_index_ort = ort.OrtValue.ortvalue_from_shape_and_type([batch_size], np.int32, "cuda", 0)
        _bind_io_output(io_binding, mask_index_name, mask_index_ort, np.int32)

    output_orts = [elm_output_ort] if one_output else [elm_output_ort, mask_index_ort]
    return (io_binding, output_orts)


def run_ait_custom_op(input_ids_np: np.ndarray, attention_mask_np: np.ndarray, token_type_np: np.ndarray):
    sess_options = ort.SessionOptions()
    shared_lib = os.path.join(ait_build_folder, model_name, "test.so")
    sess_options.register_custom_ops_library(shared_lib)
    session = ort.InferenceSession(converted_model, sess_options=sess_options, providers=["CUDAExecutionProvider"])
    (io_binding, output_orts) = ort_io_binding(session, input_ids_np, attention_mask_np, token_type_np, one_output=True, no_attention_mask=True)

    session.run_with_iobinding(io_binding)
    eln_output = output_orts[0].numpy()
    return eln_output


def run_onnx_original(input_ids_np: np.ndarray, attention_mask_np: np.ndarray, token_type_np: np.ndarray):
    model = "elm_model.onnx"
    session = ort.InferenceSession(model, providers=["CUDAExecutionProvider"])
    (io_binding, output_orts) = ort_io_binding(session, input_ids_np, attention_mask_np, token_type_np, one_output=False, no_attention_mask=False)

    session.run_with_iobinding(io_binding)
    eln_output = output_orts[0].numpy()
    mask_index_output = output_orts[1].numpy()
    return eln_output

def run_pt(input_ids_np: np.ndarray, attention_mask_np: np.ndarray, token_type_np: np.ndarray):
    # need to obtain the weights
    model = onnx.load_model(converted_model)
    weights = {}
    from onnx import numpy_helper
    for init in model.graph.initializer:
        weights[init.name] = numpy_helper.to_array(init)

    # model = layernorm(input_emb + type_emb + pos_emb)
    import torch
    import torch.nn as nn
    input_emb_op = nn.Embedding(vocab_size, embed_size, _weight = torch.from_numpy(weights["word_embedding"])).cuda().half()
    type_emb_op = nn.Embedding(2, embed_size, _weight = torch.from_numpy(weights["token_type_embedding"])).cuda().half()
    pos_emb_op = nn.Embedding(seq_len, embed_size, _weight = torch.from_numpy(weights["pos_embedding"])).cuda().half()

    ln_op = nn.LayerNorm((hidden_size, ), eps=epsilon, elementwise_affine=True).cuda().half()
    ln_op.weight = nn.Parameter(torch.from_numpy(weights["ln_weight"]).cuda().half(), requires_grad=False)
    ln_op.bias = nn.Parameter(torch.from_numpy(weights["ln_bias"]).cuda().half(), requires_grad=False)

    # run the model
    # setup inputs
    input_ids = torch.from_numpy(input_ids_np).cuda()
    token_type_ids = torch.from_numpy(token_type_np).cuda()
    pos_ids = torch.from_numpy(weights["___default_pos_ids"]).cuda()

    input_emb = input_emb_op(input_ids)
    token_type_emb = type_emb_op(token_type_ids)
    pos_emb = pos_emb_op(pos_ids)

    emb = input_emb + token_type_emb + pos_emb
    out = ln_op(emb)
    return out.detach().cpu().numpy()


def generate_inputs():
    np.random.seed(42)
    input_ids_np = np.random.randint(1, vocab_size, size=(batch_size, seq_len), dtype=np.int32)

    # attention mask: int32[batch_size, seq_len] -- all ones for now since this is ignored in AIT
    attention_mask_np = np.zeros_like(input_ids_np, dtype=np.int32)
    # token_type_ids: to identify segments of the input sequence. Let's do all zeros for now
    token_type_np = np.zeros_like(input_ids_np, dtype=np.int32)
    return (input_ids_np, attention_mask_np, token_type_np)


@click.command()
@click.option("--build", is_flag = True)
@click.option("--run", is_flag = True)
def _run(build: bool, run: bool):
    if build and run:
        logging.info("Please specify only one --build or --run")
    elif build:
        logging.info("Building custom op...")
        build_ait_custom_op()

    elif run:
        logging.info("Running to check parity...")

        # run on the same inputs and check parity
        (input_ids_np, attention_mask_np, token_type_np) = generate_inputs()
        pt_out = run_pt(input_ids_np, attention_mask_np, token_type_np)
        original_output = run_onnx_original(input_ids_np, attention_mask_np, token_type_np)
        custom_op_output = run_ait_custom_op(input_ids_np, attention_mask_np, token_type_np)
        if np.allclose(original_output, custom_op_output, atol=0.1):
            print("Outputs matched!")
        else:
            print("Outputs don't match!")
            print(f"onnx original output:\n{original_output}")
            print(f"onnx custom op output:\n{custom_op_output}")
            print(f"pt output:\n{pt_out}")

    else:
        logging.info("Please specify either --build or --run")
    

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    _run()

