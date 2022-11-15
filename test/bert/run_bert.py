from common import *
import onnxruntime as ort
import numpy as np
import logging


shared_lib = "./tmp/bert/test.so"
model_path = "bert_converted.onnx"
original_model_path = "/work/models/bert_base/onnx_models/bert_base_cased_3_fp16_gpu.onnx"


def run_onnx(custom_op: bool):
    input_dtype = np.int32 if custom_op else np.int64
    input_ids_np, attention_mask_np, token_type_np = generate_inputs(input_dtype)
    input_ids_ort = ort.OrtValue.ortvalue_from_numpy(input_ids_np, "cuda", 0)
    attention_mask_ort = ort.OrtValue.ortvalue_from_numpy(attention_mask_np, "cuda", 0)
    token_type_ort = ort.OrtValue.ortvalue_from_numpy(token_type_np, "cuda", 0)

    if custom_op:
        sess_options = ort.SessionOptions()
        sess_options.register_custom_ops_library(shared_lib)
        session = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CUDAExecutionProvider"])
    else:
        session = ort.InferenceSession(original_model_path, providers=["CUDAExecutionProvider"])
    
    output1_name = session.get_outputs()[0].name
    output2_name = session.get_outputs()[1].name

    input_dict = {
        "input_ids": input_ids_ort,
        "token_type_ids": token_type_ort
    }

    if not custom_op:
        input_dict["attention_mask"] = attention_mask_ort
    
    outputs = session.run_with_ort_values([output1_name, output2_name], input_dict)

    output1 = outputs[0].numpy()
    output2 = outputs[1].numpy()
    return (output1, output2)

def generate_inputs(dtype):
    # setup inputs
    # input ids: int32[batch_size, seq_len]
    np.random.seed(42)
    input_ids_np = np.random.randint(1, vocab_size, size=(batch_size, seq_len), dtype=dtype)
    # attention mask: int32[batch_size, seq_len] -- all ones (i.e., no mask) for now since this is ignored in AIT
    attention_mask_np = np.ones_like(input_ids_np, dtype=dtype)
    # token_type_ids: to identify segments of the input sequence. Let's do all zeros for now
    token_type_np = np.zeros_like(input_ids_np, dtype=dtype)
    return (input_ids_np, attention_mask_np, token_type_np)

def _run():
    (out1_ait, out2_ait) = run_onnx(custom_op=True)
    (out1_ort, out2_ort) = run_onnx(custom_op=False)

    if np.allclose(out2_ait, out2_ort, atol=0.1) and np.allclose(out1_ait, out1_ort, atol=0.1):
        logging.info("Outputs matched, success!")
    else:
        logging.info("Outputs doesn't match, time to debug!")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    _run()