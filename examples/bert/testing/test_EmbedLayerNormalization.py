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
# seq_len = 2
# hidden_size = 8
# embed_size = 8
# vocab_size = 4
# token_types = 2

ait_build_folder = "./tmp/"
model_name = "ELN_test"
converted_model = "eln_converted.onnx"
model_path = "eln_model.onnx"


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

    def _make_init(name: str, shape: list, zero = False):
        np_data = np.random.rand(*shape).astype(dtype)
        if zero:
            np_data = np.zeros_like(np_data)
        return helper.make_tensor(name, onnx_dtype, np_data.shape, np_data.tobytes(), raw=True)

    word_embedding_init = _make_init(input_names[2], [vocab_size, embed_size])
    pos_embedding_init = _make_init(input_names[3], [seq_len, embed_size], True)
    token_type_embedding_init = _make_init(input_names[4], [token_types, embed_size], True)
    ln_weight_init = _make_init(input_names[5], [hidden_size])
    ln_bias_init = _make_init(input_names[6], [hidden_size])
    inits = [word_embedding_init, pos_embedding_init, token_type_embedding_init, ln_weight_init, ln_bias_init]
    
    # embed_out is before output before layer norm (see onnxruntime/core/graph/contrib_ops/bert_defs.cc)
    output_names = ["eln_output", "mask_index", "embed_out"]
    node = helper.make_node(
        op_type="EmbedLayerNormalization",
        inputs = input_names,
        outputs = output_names,
        name="eln",
        domain="com.microsoft",
        epsilon=epsilon
    )

    # outputs
    eln_output = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT16, [batch_size, seq_len, hidden_size])
    mask_index_output = helper.make_tensor_value_info(output_names[1], TensorProto.INT32, [batch_size])
    embed_out = helper.make_tensor_value_info(output_names[2], TensorProto.FLOAT16, [batch_size, seq_len, hidden_size])
    outputs = [eln_output, mask_index_output, embed_out]

    graph = helper.make_graph([node], "eln-graph", inputs, outputs, inits)
    model = helper.make_model(graph)
    onnx.save_model(model, model_path)
    return model


def build_ait_custom_op():
    from converter import transform_graph, compile
    from custom_op_generator import generate, convert_graph
    
    model = create_graph()

    # remove mask_index and embed_out outputs since it's not an output in AIT
    model.graph.output.pop()
    model.graph.output.pop()
    model.graph.node[0].output.pop()
    model.graph.node[0].output.pop()

    attributes={
        "batch_size": batch_size,
        "hidden_size": hidden_size, 
        "seq_len": seq_len}

    context = compile(model, ait_build_folder, model_name, attributes=attributes)
    generate(context, os.path.join(ait_build_folder, model_name))
    convert_graph(model.graph, context, converted_model)


def run_onnx(custom_op: bool):
    (input_ids_np, attention_mask_np, token_type_np) = generate_inputs()
    input_ids_ort = ort.OrtValue.ortvalue_from_numpy(input_ids_np, "cuda", 0)
    attention_mask_ort = ort.OrtValue.ortvalue_from_numpy(attention_mask_np, "cuda", 0)
    token_type_ort = ort.OrtValue.ortvalue_from_numpy(token_type_np, "cuda", 0)

    if custom_op:
        sess_options = ort.SessionOptions()
        shared_lib = os.path.join(ait_build_folder, model_name, "test.so")
        sess_options.register_custom_ops_library(shared_lib)
        session = ort.InferenceSession(converted_model, sess_options=sess_options, providers=["CUDAExecutionProvider"])
    else:
        session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])

    eln_output_name = session.get_outputs()[0].name
    assert(eln_output_name == "eln_output")

    output_names = [eln_output_name]
    if not custom_op:
        # original graph has two extra outputs
        mask_index_output_name = session.get_outputs()[1].name
        embed_out_name = session.get_outputs()[2].name
        
        assert(mask_index_output_name == "mask_index")
        assert(embed_out_name == "embed_out")
        output_names += [mask_index_output_name, embed_out_name]

    input_dict = {
        "input_ids" : input_ids_ort,
        "token_type_ids": token_type_ort
    }

    if not custom_op:
        # original graph takes in extra input, attention_mask
        input_dict["attention_mask"] = attention_mask_ort

    outputs = session.run_with_ort_values(output_names, input_dict)

    eln_output = outputs[0].numpy()
    if not custom_op:
        mask_index = outputs[1].numpy()
        embed_output = outputs[2].numpy()

    # only eln_output is used for parity check
    return eln_output


def run_pt():
    (input_ids_np, attention_mask_np, token_type_np) = generate_inputs()

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
    type_emb_op = nn.Embedding(token_types, embed_size, _weight = torch.from_numpy(weights["token_type_embedding"])).cuda().half()
    pos_emb_op = nn.Embedding(seq_len, embed_size, _weight = torch.from_numpy(weights["pos_embedding"])).cuda().half()

    ln_op = nn.LayerNorm(hidden_size, eps=epsilon, elementwise_affine=False).cuda().half()
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
    input_ids_np[0][0] = 2
    input_ids_np[0][1] = 3

    # attention_mask doesn't have any influence in the EmberLayerNormalization calculation
    # only used to produce mask_index which is the first zero value of each sentence 
    # (i.e., lenght of actual sentence without padding) 
    attention_mask_np = np.ones_like(input_ids_np, dtype=np.int32)

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

        pt_out = run_pt()
        original_output = run_onnx(custom_op=False)
        custom_op_output = run_onnx(custom_op=True)
        if np.allclose(original_output, custom_op_output, atol=0.1) and np.allclose(pt_out, custom_op_output, atol=0.1):
            print("Outputs matched!")
        else:
            print("Outputs don't match!")
            print(f"pt output:\n{pt_out}")
            print(f"onnx original output:\n{original_output}")
            print(f"onnx custom op output:\n{custom_op_output}")
    else:
        logging.info("Please specify either --build or --run")
    

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    _run()

