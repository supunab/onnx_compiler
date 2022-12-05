from common import *
import onnxruntime as ort
import numpy as np
import logging
import torch.utils.benchmark as benchmark
import time
import click

shared_lib = "/work/onnx_compiler/examples/distilgpt2/tmp/distilgpt2/test.so"
# shared_lib = "/work/supun/onnx_compiler/examples/distilgpt2/tmp/distilgpt2/test.so"
model_path = "distilgpt2_converted.onnx"
original_model_path = "/work/models/distilgpt2/distilgpt2_fp16.onnx"
#original_model_path = "/work/supun/models/distilgpt2/distilgpt2_fp16.onnx"


def run_ait(bench: bool, config: dict):
    from aitemplate.compiler import Model
    import onnx
    from onnx import numpy_helper

    batch_size = config["batch_size"]
    hidden_size = config["hidden_size"]
    past_seq_len = config["past_seq_len"]
    curr_seq_len = config["seq_len"]
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
    input_ids_np, position_ids_np, attention_mask_np, past_list = generate_inputs(input_dtype, config)

    input_ids_ait = mod.numpy_to_ait_data(input_ids_np)
    position_ids_ait = mod.numpy_to_ait_data(position_ids_np)

    inputs = {
        "input_ids": input_ids_ait,
        "position_ids": position_ids_ait
    }
    for i in range(num_layers):
        inputs[f"past_{i}"] = mod.numpy_to_ait_data(past_list[i])

    # outputs
    output_logits = mod.numpy_to_ait_data(np.empty([batch_size, curr_seq_len, hidden_size], dtype=np.float16))
    output_present_list = []
    for i in range(num_layers):
        output_present_list.append(mod.numpy_to_ait_data(np.empty([2, batch_size, num_heads_default, curr_seq_len, hidden_size // num_heads_default], dtype=np.float16)))

    outputs = {
        "logits": output_logits
    }
    for i in range(num_layers):
        outputs[f"present_{i}"] = output_present_list[i]

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

    output_logits_np = mod.ait_data_to_numpy(output_logits)
    output_presents_np = list(map(lambda x: mod.ait_data_to_numpy(x), output_present_list))

    return (output_logits_np, output_presents_np, iter_time) 


def run_onnx(custom_op: bool, bench: bool, config: dict):
    logging.info(f"""Running {"AIT custom op" if custom_op else "ONNX original"}....""")
    input_dtype = np.int32 if custom_op else np.int64
    input_ids_np, position_ids_np, attention_mask_np, past_list = generate_inputs(input_dtype, config)

    input_ids_ort = ort.OrtValue.ortvalue_from_numpy(input_ids_np, "cuda", 0)
    attention_mask_ort = ort.OrtValue.ortvalue_from_numpy(attention_mask_np, "cuda", 0)
    position_ids_ort = ort.OrtValue.ortvalue_from_numpy(position_ids_np, "cuda", 0)
    past_orts = []
    for i in range(num_layers):
        past_orts.append(
            ort.OrtValue.ortvalue_from_numpy(past_list[i], "cuda", 0)
        )

    # provider with memory arena config
    # cuda_provider_with_options = ("CUDAExecutionProvider",
    # {'device_id': 0,
    # 'arena_extend_strategy' : 'kNextPowerOfTwo',
    # 'gpu_mem_limit' : 3 * 1024 * 1024 * 1024} # init with 3 GB memory arena
    # )

    if custom_op:
        sess_options = ort.SessionOptions()
        sess_options.register_custom_ops_library(shared_lib)
        session = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CUDAExecutionProvider"])
    else:
        session = ort.InferenceSession(original_model_path, providers=["CUDAExecutionProvider"])

    input_dict = {
        "input_ids": input_ids_ort,
        "position_ids": position_ids_ort
    }
    for i in range(num_layers):
        input_dict[f"past_{i}"] = past_orts[i]

    # attention mask is an input in the original model
    if not custom_op:
        input_dict["attention_mask"] = attention_mask_ort

    output_logits_name = session.get_outputs()[0].name
    output_present_names = list(map(lambda x: x.name, session.get_outputs()[1:]))
    
    output_names = [output_logits_name] + output_present_names

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

    output_logits = outputs[0].numpy()
    output_present_list = list(map(lambda x: x.numpy(), outputs[1:]))

    return (output_logits, output_present_list, iter_time)


def generate_inputs(dtype, config):
    vocab_size = config["vocab_size"]
    batch_size = config["batch_size"]
    past_seq_len = config["past_seq_len"]
    curr_seq_len = config["seq_len"]
    hidden_size = config["hidden_size"]

    # set seed
    np.random.seed(42)

    # input_ids int32[batch_size, seq_len]
    input_ids_np = np.random.randint(1, vocab_size, size=(batch_size, curr_seq_len), dtype=np.int32).astype(dtype)
    # position_ids int32[batch_size, seq_len]
    position_ids_np = np.tile(np.arange(past_seq_len, past_seq_len + curr_seq_len), batch_size).reshape(batch_size, -1).astype(dtype)
    # attention mask: int32[batch_size, seq_len] -- all ones (i.e., no mask) for now since this is ignored in AIT
    attention_mask_np = np.ones_like(input_ids_np, dtype=np.int32).astype(dtype)

    # need to generate past0, past1, etc. : float16[batch, past_seq_len, hidden_size]
    past_list = []
    for _ in range(num_layers):
        past_list.append(np.random.rand(2, batch_size, num_heads_default, past_seq_len, hidden_size // num_heads_default).astype(np.float16))

    return (input_ids_np, position_ids_np, attention_mask_np, past_list)


def run_gpt2(run_original: bool, run_custom: bool, run_ait_generated: bool, benchmark: bool, compare: bool, batch_size: int, hidden_size: int, past_seq_len: int, curr_seq_len: int, vocab_size: int):
    config = {
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "past_seq_len": past_seq_len,
        "seq_len": curr_seq_len,
        "total_seq_len": past_seq_len + curr_seq_len,
        "vocab_size": vocab_size,
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

    elif compare:
        (custom_logits_out, custom_present_list, _) = run_onnx(custom_op=True, bench=benchmark, config=config)
        (ort_logits_out, ort_present_list, _) = run_onnx(custom_op=False, bench=benchmark, config=config)
        # (out1_ait, out2_ait) = run_ait()  Note - manually verified the output is the same

        if np.allclose(custom_logits_out, ort_logits_out, atol=0.1):
            logging.info("Output logits matched, success!")
        else:
            logging.info("Output logits doesn't match, time to debug!")

        for i in range(num_layers):
            if np.allclose(custom_present_list[i], ort_present_list[i], atol=0.1):
                logging.info(f"Output present_{i} matched, success!")
            else:
                logging.info(f"Output present_{i} doesn't match!")
    else:
        logging.info("Please specify --run_original --run_custom --run_ait_generated --benchmark")


@click.command()
@click.option("--run_original", is_flag=True)
@click.option("--run_custom", is_flag=True)
@click.option("--run_ait_generated", is_flag=True)
@click.option("--benchmark", is_flag=True)
@click.option("--compare", is_flag=True)
@click.option("--batch_size", default=batch_size_default)
@click.option("--hidden_size", default=hidden_size_default)
@click.option("--past_seq_len", default=past_seq_len_default)
@click.option("--curr_seq_len", default=curr_seq_len_default)
@click.option("--vocab_size", default=vocab_size_default)
def _run_gpt2(run_original: bool, run_custom: bool, run_ait_generated: bool, benchmark: bool, compare: bool, batch_size: int, hidden_size: int, past_seq_len: int, curr_seq_len: int, vocab_size: int):
    run_gpt2(run_original, run_custom, run_ait_generated, benchmark, compare, batch_size, hidden_size, past_seq_len, curr_seq_len, vocab_size)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    _run_gpt2()