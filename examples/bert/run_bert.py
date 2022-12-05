from common import *
import onnxruntime as ort
import numpy as np
import logging
import torch.utils.benchmark as benchmark
import time
import click

# shared_lib = "/work/supun/onnx_compiler/test/bert/tmp/bert/test.so"
shared_lib = "/work/onnx_compiler/examples/bert/tmp/bert/test.so"
model_path = "bert_converted.onnx"
original_model_path = "/work/models/bert_base/onnx_models/bert_base_cased_3_fp16_gpu.onnx"
# original_model_path = "/work/supun/models/bert/bert_base_cased_3_fp16_gpu.onnx"


def _benchmark_ort(session, output_names, input_dict):
    session.run_with_ort_values(output_names, input_dict)

def run_ait(bench: bool, config: dict):
    from aitemplate.compiler import Model
    import onnx
    from onnx import numpy_helper

    batch_size = config["batch_size"]
    hidden_size = config["hidden_size"]
    seq_len = config["seq_len"]
    logging.info("Running generated AIT code directly....")

    # load the model
    model = onnx.load_model(model_path)
    graph = model.graph

    # load the compiled .so
    mod = Model(shared_lib)

    # set constants
    for init in graph.initializer:
        name = init.name
        np_data = numpy_helper.to_array(init)
        ait_data = mod.numpy_to_ait_data(np_data)
        mod.set_constant(name, ait_data)
    

    # setup inputs
    input_dtype = np.int32
    input_ids_np, attention_mask_np, token_type_np = generate_inputs(input_dtype, config)

    input_ids_ait = mod.numpy_to_ait_data(input_ids_np)
    token_type_ait = mod.numpy_to_ait_data(token_type_np)

    inputs = {
        "input_ids": input_ids_ait,
        "token_type_ids": token_type_ait
    }

    # outputs
    out1_ait = mod.numpy_to_ait_data(np.empty([batch_size, seq_len, hidden_size], dtype=np.float16))
    out2_ait = mod.numpy_to_ait_data(np.empty([batch_size, hidden_size], dtype=np.float16))

    outputs = {
        "graph_output_cast_0": out1_ait,
        "graph_output_cast_1": out2_ait
    }

    iter_time = 0
    if bench:
        # warm ups
        for i in range(warm_ups):
            mod.run(inputs, outputs, sync=True)
        
        start = time.time()
        for i in range(repeats):
            mod.run(inputs, outputs, sync=True)
        end = time.time()
        iter_time = (end - start) / repeats * 1000
        logging.info(f"Elapsed Time: {iter_time}ms")
    else:
        start = time.time()
        mod.run(inputs, outputs, sync=True)
        end = time.time()
        iter_time = (end - start) * 1000
        logging.info(f"Elapsed Time: {iter_time}ms")

    return (mod.ait_data_to_numpy(out1_ait), mod.ait_data_to_numpy(out2_ait), iter_time) 


def run_onnx(custom_op: bool, bench: bool, config: dict):
    logging.info(f"""Running {"AIT custom op" if custom_op else "ONNX original"}....""")
    input_dtype = np.int32 if custom_op else np.int64
    input_ids_np, attention_mask_np, token_type_np = generate_inputs(input_dtype, config)
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

    iter_time = 0
    if bench:
        # warm ups
        for i in range(warm_ups):
            outputs = session.run_with_ort_values(output_names, input_dict)
        
        start = time.time()
        for i in range(repeats):
            outputs = session.run_with_ort_values(output_names, input_dict)
        end = time.time()
        iter_time = (end - start) / repeats * 1000
        logging.info(f"Elapsed Time: {iter_time}ms")
    else:
        start = time.time()
        outputs = session.run_with_ort_values(output_names, input_dict)
        end = time.time()
        iter_time = (end - start)* 1000
        logging.info(f"Elapsed Time: {iter_time}ms")
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

    return (output1, output2, iter_time)

def generate_inputs(dtype, config):
    vocab_size = config["vocab_size"]
    batch_size = config["batch_size"]
    seq_len = config["seq_len"]
    # setup inputs
    # input ids: int32[batch_size, seq_len]
    np.random.seed(42)
    input_ids_np = np.random.randint(1, vocab_size, size=(batch_size, seq_len), dtype=dtype)
    # attention mask: int32[batch_size, seq_len] -- all ones (i.e., no mask) for now since this is ignored in AIT
    attention_mask_np = np.ones_like(input_ids_np, dtype=dtype)
    # token_type_ids: to identify segments of the input sequence. Let's do all zeros for now
    token_type_np = np.zeros_like(input_ids_np, dtype=dtype)
    return (input_ids_np, attention_mask_np, token_type_np)

def run_bert(run_original: bool, run_custom: bool, run_ait_generated: bool, benchmark: bool, batch_size: int, hidden_size: int, seq_len: int, vocab_size: int):
    config = {
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "seq_len": seq_len,
        "vocab_size": vocab_size
    }

    if run_custom:
        (_, _, iter_time) = run_onnx(custom_op=True, bench=benchmark, config=config)
        return iter_time

    elif run_original:
        (_, _, iter_time) = run_onnx(custom_op=False, bench=benchmark, config=config)
        return iter_time

    elif run_ait_generated:
        (_, _, iter_time) = run_ait(bench=benchmark, config=config)
        return iter_time

    elif benchmark:
        (out1_custom, out2_custom, _) = run_onnx(custom_op=True, bench=True, config=config)
        (out1_ort, out2_ort, _) = run_onnx(custom_op=False, bench=True, config=config)
        # (out1_ait, out2_ait) = run_ait()  Note - manually verified the output is the same

        if np.allclose(out2_custom, out2_ort, atol=0.1) and np.allclose(out1_custom, out1_ort, atol=0.1):
            logging.info("Outputs matched, success!")
        else:
            logging.info("Outputs doesn't match, time to debug!")
    else:
        logging.info("Please specify --run_original --run_custom --run_ait_generated --benchmark")


@click.command()
@click.option("--run_original", is_flag=True)
@click.option("--run_custom", is_flag=True)
@click.option("--run_ait_generated", is_flag=True)
@click.option("--benchmark", is_flag=True)
@click.option("--batch_size", default=batch_size_default)
@click.option("--hidden_size", default=hidden_size_default)
@click.option("--seq_len", default=seq_len_default)
@click.option("--vocab_size", default=vocab_size_default)
def _run_bert(run_original: bool, run_custom: bool, run_ait_generated: bool, benchmark: bool, batch_size: int, hidden_size: int, seq_len: int, vocab_size: int):
    run_bert(run_original, run_custom, run_ait_generated, benchmark, batch_size, hidden_size, seq_len, vocab_size)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    _run_bert()