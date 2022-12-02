"""
Unit test for Attention in GPT2. This is different from BERT for several reasons.
1) GPT2 attention is unidirectional. Meaning that a token can only attend to previous tokens
2) May use previous state as an input instead of computing attention for old tokens again (to prevent recomputing the same thing)
"""

from __future__ import annotations
import sys
import os

sys.path.insert(1, os.path.abspath("./../../../"))
import numpy as np
import onnx
import click

"""
Past states support
Unfused version : doable
flash attention : not supported in the AITFrontend (it takes a single projected qkv), how about the actual FlashAttention kernel?
mem_eff_attention : need to check, if takes q, k, v separately, so it might be possible?
"""

"""
Unidirectional attention (for sequence length = 1, it doesn't matter)
unfused : doable
flash attention : causal = True? (see ops.flash_attentio(causal=True))
mem_eff_attention : causal = True would do that?
"""

from common import *

model_path = "attn_model.onnx"
converted_model_path = "attn_model_converted.onnx"
def create_onnx_graph():
    from onnx import helper
    from onnx import TensorProto
    input_np, past0_np = generate_inputs()
    qkv_weight_np, qkv_bias_np = generate_weights()
    
    input_names = ["input_hidden_states", "qkv_weight", "qkv_bias", "attn_mask", "past0"]
    input_hidden_states = helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT16, input_np.shape)
    past0 = helper.make_tensor_value_info(input_names[-1], TensorProto.FLOAT16, past0_np.shape)
    inputs = [input_hidden_states, past0]

    # initializers
    qkv_weight = helper.make_tensor(input_names[1], TensorProto.FLOAT16, qkv_weight_np.shape, qkv_weight_np.tobytes(), raw=True)
    qkv_bias = helper.make_tensor(input_names[2], TensorProto.FLOAT16, qkv_bias_np.shape, qkv_bias_np.tobytes(), raw=True)
    attention_mask_np = np.ones([batch_size, prev_seq_len + curr_seq_len], dtype=np.int32)
    attention_mask = helper.make_tensor(input_names[3], TensorProto.INT32, attention_mask_np.shape, attention_mask_np.tobytes(), raw=True)
    inits = [qkv_weight, qkv_bias, attention_mask]

    output_names = ["attention_out", "present0"]
    attention_out = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT16, [batch_size, curr_seq_len, hidden_size])
    present0 = helper.make_tensor_value_info(output_names[1], TensorProto.FLOAT16, [2, batch_size, num_heads, prev_seq_len + curr_seq_len, hidden_size // num_heads])
    outputs = [attention_out, present0]

    node = helper.make_node(
        op_type="Attention",
        inputs=input_names,
        outputs=output_names,
        name="atn",
        domain="com.microsoft",
        num_heads=num_heads,
        unidirectional=True
    )

    graph = helper.make_graph([node], "attn_unidirectional_graph", inputs, outputs, inits)
    model = helper.make_model(graph)
    onnx.save_model(model, model_path)
    return model

def build_custom_op():
    from converter import transform_graph, compile
    from custom_op_generator import generate, convert_graph

    model = onnx.load_model(model_path)
    attributes = {
        "batch_size": batch_size,
        "hidden_size": hidden_size, 
        "past_seq_len": prev_seq_len,
        "curr_seq_len": curr_seq_len }

    context = compile(model, model_name="mem_eff_attn", attributes=attributes)
    generate(context, "./tmp/mem_eff_attn/")
    convert_graph(model.graph, context, converted_model_path)


def run_ort(custom: bool):
    import onnxruntime as ort
    input_np, past0_np = generate_inputs()
    input_ort = ort.OrtValue.ortvalue_from_numpy(input_np, "cuda", 0)
    past0_ort = ort.OrtValue.ortvalue_from_numpy(past0_np, "cuda", 0)

    session = None
    cuda_provider_with_options = ("CUDAExecutionProvider",
    {'device_id': 0,
    'arena_extend_strategy' : 'kSameAsRequested',
    'gpu_mem_limit' : 1024 * 1024 * 1024} # init with 1 GB memory arena
    )
    if custom:
        sess_options = ort.SessionOptions()
        shared_lib = "./tmp/mem_eff_attn/test.so"
        sess_options.register_custom_ops_library(shared_lib)
        session = ort.InferenceSession(converted_model_path, sess_options=sess_options, providers=[cuda_provider_with_options])
    else:
        session = ort.InferenceSession(model_path, providers=[cuda_provider_with_options])

    io_binding = session.io_binding()
    input0_name = session.get_inputs()[0].name
    # io_binding.bind_ortvalue_input(input0_name, input_ort)
    io_binding.bind_input(input0_name, "cuda", 0, np.float16, input_ort.shape(), input_ort.data_ptr())
    input1_name = session.get_inputs()[1].name
    # io_binding.bind_ortvalue_input(input1_name, past0_ort)
    io_binding.bind_input(input1_name, "cuda", 0, np.float16, past0_ort.shape(), past0_ort.data_ptr())

    # inputs = {
    #     "input_hidden_states": input_ort,
    #     "past0": past0_np
    # }
    # outputs = session.run_with_ort_values(["attention_out", "present0"], inputs)
    # attention_out = outputs[0].numpy()
    # present0 = outputs[1].numpy()
    # return (attention_out, present0)

    attn_out_ort = ort.OrtValue.ortvalue_from_shape_and_type([batch_size, curr_seq_len, hidden_size], np.float16, "cuda", 0)
    present0_ort = ort.OrtValue.ortvalue_from_shape_and_type([2, batch_size, num_heads, prev_seq_len + curr_seq_len, hidden_size // num_heads], np.float16, "cuda", 0)
    output0_name = session.get_outputs()[0].name
    io_binding.bind_ortvalue_output(output0_name, attn_out_ort)
    output1_name = session.get_outputs()[1].name
    io_binding.bind_ortvalue_output(output1_name, present0_ort)
    
    session.run_with_iobinding(iobinding=io_binding)
    # we have to sync otherwise, we don't have the final result
    return (attn_out_ort.numpy(), present0_ort.numpy())


from ait_testAttention import mem_eff_attention_unidirectional

@click.command()
@click.option("--create_onnx", is_flag=True)
@click.option("--build_custom", is_flag=True)
@click.option("--run_custom", is_flag=True)
@click.option("--compare", is_flag=True)
def _run(create_onnx: bool, build_custom: bool, run_custom: bool, compare: bool):
    if create_onnx:
        create_onnx_graph()
    
    if build_custom:
        build_custom_op()

    if run_custom:
        out0, out1 = run_ort(custom=True)
        print(out1)
        print(out0)

    if compare:
        ort_out0, ort_out1 = run_ort(custom=False)
        ait_out0, ait_out1 = run_ort(custom=True)
        # ait_out0, ait_out1 = mem_eff_attention_unidirectional()

        if np.allclose(ort_out1, ait_out1, atol=0.1):
            print("output 1 matched")
        else:
            print("output 1 doesn't match")
            print("ORT")
            print(ort_out1)
            print("AIT")
            print(ait_out1)

        if np.allclose(ort_out0, ait_out0, atol=0.1):
            print("output 0 matched")
        else:
            print("output 0 doesn't match")
            print("ORT")
            print(ort_out0)
            print("AIT")
            print(ait_out0)

if __name__ == "__main__":
    _run()