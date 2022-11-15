from common import *
import onnxruntime as ort
import numpy as np
import logging
import torch.utils.benchmark as benchmark
import time
import click

shared_lib = "./tmp/bert/test.so"
model_path = "bert_converted.onnx"
original_model_path = "/work/models/bert_base/onnx_models/bert_base_cased_3_fp16_gpu.onnx"


def _benchmark_ort(session, output_names, input_dict):
    session.run_with_ort_values(output_names, input_dict)

def run_onnx(custom_op: bool, bench: bool):
    logging.info(f"""Running {"AIT custom op" if custom_op else "ONNX original"}....""")
    input_dtype = np.int32 if custom_op else np.int64
    input_ids_np, attention_mask_np, token_type_np = generate_inputs(input_dtype)
    input_ids_ort = ort.OrtValue.ortvalue_from_numpy(input_ids_np, "cuda", 0)
    attention_mask_ort = ort.OrtValue.ortvalue_from_numpy(attention_mask_np, "cuda", 0)
    token_type_ort = ort.OrtValue.ortvalue_from_numpy(token_type_np, "cuda", 0)

    # provider with memory arena config
    cuda_provider_with_options = ("CUDAExecutionProvider",
    {'device_id': 0,
    'arena_extend_strategy' : 'kNextPowerOfTwo',
    'gpu_mem_limit' : 3 * 1024 * 1024 * 1024} # init with 3 GB memory arena
    )

    if custom_op:
        sess_options = ort.SessionOptions()
        sess_options.register_custom_ops_library(shared_lib)
        session = ort.InferenceSession(model_path, sess_options=sess_options, providers=[cuda_provider_with_options])
    else:
        session = ort.InferenceSession(original_model_path, providers=[cuda_provider_with_options])
    
    output1_name = session.get_outputs()[0].name
    output2_name = session.get_outputs()[1].name

    input_dict = {
        "input_ids": input_ids_ort,
        "token_type_ids": token_type_ort
    }

    if not custom_op:
        input_dict["attention_mask"] = attention_mask_ort
    
    output_names = [output1_name, output2_name]
    start = time.time()
    iter = 1
    for i in range(iter):
        outputs = session.run_with_ort_values(output_names, input_dict)
    end = time.time()
    logging.info(f"Elapsed Time: {end - start}")
    output1 = outputs[0].numpy()
    output2 = outputs[1].numpy()

    # do benchmarking
    # if bench:
    #     timer = benchmark.Timer(
    #         stmt="_benchmark_ort(session, output_names, input_dict)",
    #         setup="from __main__ import _benchmark_ort",
    #         globals={"session": session, "output_names" : output_names, "input_dict": input_dict}
    #     )
    #     timer.timeit(warm_ups) # warm up
    #     print(f"""{"ORT (AIT Custom Op)" if custom_op else "ORT Original"}: {timer.timeit(repeats)}""")

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

@click.command()
@click.option("--run_original", is_flag=True)
@click.option("--run_custom", is_flag=True)
@click.option("--benchmark", is_flag=True)
def _run(run_original: bool, run_custom: bool, benchmark: bool):
    if run_custom:
        run_onnx(custom_op=True, bench=False)

    if run_original:
        run_onnx(custom_op=False, bench=False)

    if benchmark:
        (out1_ait, out2_ait) = run_onnx(custom_op=True, bench=True)
        (out1_ort, out2_ort) = run_onnx(custom_op=False, bench=True)

        if np.allclose(out2_ait, out2_ort, atol=0.1) and np.allclose(out1_ait, out1_ort, atol=0.1):
            logging.info("Outputs matched, success!")
        else:
            logging.info("Outputs doesn't match, time to debug!")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    _run()