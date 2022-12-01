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

def run_ort():
    import onnxruntime as ort
    input_np, past0_np = generate_inputs()
    input_ort = ort.OrtValue.ortvalue_from_numpy(input_np)
    past0_np = ort.OrtValue.ortvalue_from_numpy(past0_np)
    
    session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
    inputs = {
        "input_hidden_states": input_ort,
        "past0": past0_np
    }
    outputs = session.run_with_ort_values(["attention_out", "present0"], inputs)
    attention_out = outputs[0].numpy()
    present0 = outputs[1].numpy()
    # print(attention_out)
    # print(present0)
    return (attention_out, present0)


from ait_testAttention import mem_eff_attention_unidirectional
if __name__ == "__main__":
    create_onnx_graph()
    ort_out0, ort_out1 = run_ort()
    ait_out0, ait_out1 = mem_eff_attention_unidirectional()

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

