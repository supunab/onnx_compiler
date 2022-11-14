from common import *
import onnxruntime as ort
import numpy as np
import logging


# def run_pytorch(input_ids_np: np.ndarray, attention_mask_np: np.ndarray, token_type_np: np.ndarray):
#     from transformers import AutoModelForMaskedLM

    

def _bind_io_input(io_binding: ort.IOBinding, name: str, data_np: np.ndarray)->None:
    data_ort = ort.OrtValue.ortvalue_from_numpy(data_np, "cuda", 0)
    io_binding.bind_input(name=name, device_type=data_ort.device_name(), device_id=0, element_type=data_np.dtype,
                            shape=data_ort.shape(), buffer_ptr=data_ort.data_ptr())

def _bind_io_output(io_binding: ort.IOBinding, name: str, ort_value: ort.OrtValue, elem_type)->None:
    io_binding.bind_output(name=name, device_type=ort_value.device_name(), device_id=0, element_type=elem_type,
                            shape=ort_value.shape(), buffer_ptr=ort_value.data_ptr())

def run_onnx_model_original(model_path: str, input_ids_np: np.ndarray, attention_mask_np: np.ndarray, token_type_np: np.ndarray):
    logging.info("Running original ONNX model")
    input_ids_np = input_ids_np.astype(np.int64)
    attention_mask_np = attention_mask_np.astype(np.int64)
    token_type_np = token_type_np.astype(np.int64)
    
    sess_options = ort.SessionOptions()
    session = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CUDAExecutionProvider"])
    io_binding = session.io_binding()
    input_ids_name = session.get_inputs()[0].name
    _bind_io_input(io_binding, input_ids_name, input_ids_np)

    attention_mask_name = session.get_inputs()[1].name
    _bind_io_input(io_binding, attention_mask_name, attention_mask_np)

    token_type_name = session.get_inputs()[2].name
    _bind_io_input(io_binding, token_type_name, token_type_np)

    # original model casts outputs back to float32 from float16
    output_dtype = np.float32

    # create output ort values
    output0_name = session.get_outputs()[0].name
    output0_ort = ort.OrtValue.ortvalue_from_shape_and_type((batch_size, seq_len, hidden_size), output_dtype, "cuda", 0)
    _bind_io_output(io_binding, output0_name, output0_ort, output_dtype)

    
    output1_name = session.get_outputs()[1].name
    output1_ort = ort.OrtValue.ortvalue_from_shape_and_type((batch_size, hidden_size), output_dtype, "cuda", 0)
    _bind_io_output(io_binding, output1_name, output1_ort, output_dtype)
    session.run_with_iobinding(io_binding)
    return (output0_ort.numpy(), output1_ort.numpy())

def run_onnx_model_custom_op(model_path: str, shared_lib: str, input_ids_np: np.ndarray, attention_mask_np: np.ndarray, token_type_np: np.ndarray):
    logging.info("Running AIT custom op based ONNX model")
    sess_options = ort.SessionOptions()
    sess_options.register_custom_ops_library(shared_lib)
    session = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CUDAExecutionProvider"])
    io_binding = session.io_binding()
    input_ids_name = session.get_inputs()[0].name
    _bind_io_input(io_binding, input_ids_name, input_ids_np)

    # no attention mask in the converted model
    # attention_mask_name = session.get_inputs()[1].name
    # _bind_io_input(io_binding, attention_mask_name, attention_mask_np)

    token_type_name = session.get_inputs()[1].name
    _bind_io_input(io_binding, token_type_name, token_type_np)

    # create output ort values
    output0_name = session.get_outputs()[0].name
    output0_ort = ort.OrtValue.ortvalue_from_shape_and_type((batch_size, seq_len, hidden_size), np.float16, "cuda", 0)
    _bind_io_output(io_binding, output0_name, output0_ort, np.float16)

    output1_name = session.get_outputs()[1].name
    output1_ort = ort.OrtValue.ortvalue_from_shape_and_type((batch_size, hidden_size), np.float16, "cuda", 0)
    _bind_io_output(io_binding, output1_name, output1_ort, np.float16)
    session.run_with_iobinding(io_binding)
    return (output0_ort.numpy(), output1_ort.numpy())

def _run():
    shared_lib = "./tmp/bert/test.so"
    model_path = "bert_converted.onnx"
    original_model_path = "/work/models/bert_base/onnx_models/bert_base_cased_3_fp16_gpu.onnx"

    # setup inputs
    # input ids: int32[batch_size, seq_len]
    input_ids_np = np.random.randint(1, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
    # attention mask: int32[batch_size, seq_len] -- all ones for now since this is ignored in AIT
    attention_mask_np = np.zeros_like(input_ids_np, dtype=np.int32)
    # token_type_ids: to identify segments of the input sequence. Let's do all zeros for now
    token_type_np = np.zeros_like(input_ids_np, dtype=np.int32)

    (out0_ait, out1_ait) = run_onnx_model_custom_op(model_path, shared_lib, input_ids_np, attention_mask_np, token_type_np)
    (out0_ort, out1_ort) = run_onnx_model_original(original_model_path, input_ids_np, attention_mask_np, token_type_np)

    np.savetxt("out0_ait.txt", out0_ait[0])
    np.savetxt("out0_ort.txt", out0_ort[0])

    if np.allclose(out0_ait, out0_ort) and np.allclose(out1_ait, out1_ort):
        logging.info("Outputs matched, success!")
    else:
        logging.info("Outputs doesn't match, time to debug!")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    _run()