"""
Few synthetic benchmarks to demonstrate the efficiency of kernel fusion
1. sequence of n elementwise ops (fused)
2. gemm-fast-gelu-gemm fusion
"""
import sys
import os

sys.path.insert(1, os.path.abspath("./../../"))
import logging
import click
import numpy as np
import time

original_model_elementwise = "n_elementwise.onnx"
converted_model_elementwise = "n_elementwise_converted.onnx"
model_name = "n_elementwise"
generated_folder = "./tmp/" + model_name
shared_lib = generated_folder + "/test.so"

batch_size_default = 32
hidden_size_default = 1024
num_elementwise_default = 10

warm_ups = 30
repeats = 100


def create_graph(batch_size: int, hidden_size: int, num_elementwise: int):
    import onnx
    from onnx import helper
    from onnx import TensorProto
    assert num_elementwise > 0
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT16, [batch_size, hidden_size])
    output_tensor = helper.make_tensor_value_info(f"output_{num_elementwise - 1}", TensorProto.FLOAT16, [batch_size, hidden_size])
    nodes = []
    nodes.append(
        helper.make_node("Exp", ["input"], ["output_0"], "exp_0")
    )
    for i in range(1, num_elementwise):
        nodes.append(
            helper.make_node("Exp", [f"output_{i-1}"], [f"output_{i}"], f"exp_{i}")
        )

    graph = helper.make_graph(
        nodes, "n-elementwise", [input_tensor], [output_tensor]
    )
    model = helper.make_model(graph)
    onnx.save_model(model, "n_elementwise.onnx")
    return model


def build_custom_op(ait_path: str, onnx_path: str):
    import onnx
    from converter import transform_graph, compile, remove_attention_mask_hack
    from custom_op_generator import generate, convert_graph
    model = onnx.load_model(original_model_elementwise)
    context = compile(model, model_name=model_name)
    generate(context, folder=generated_folder, ait_path=ait_path, onnx_header_path=onnx_path)
    convert_graph(model.graph, context, converted_model_elementwise)


def generate_input(bs: int, hs: int):
    np.random.seed(42)
    return np.random.rand(bs, hs).astype(np.float16)


def run_onnx(custom: bool, benchmark: bool, bs:int, hs: int):
    import onnxruntime as ort
    input_np = generate_input(bs, hs)
    input_ort = ort.OrtValue.ortvalue_from_numpy(input_np, "cuda", 0)

    if custom:
        sess_options = ort.SessionOptions()
        sess_options.register_custom_ops_library(shared_lib)
        session = ort.InferenceSession(converted_model_elementwise, sess_options=sess_options, providers=["CUDAExecutionProvider"])
    else:
        session = ort.InferenceSession(original_model_elementwise, providers=["CUDAExecutionProvider"])

    input_dict = {
        "input": input_ort
    }

    output_name = session.get_outputs()[0].name
    iter_time = 0
    if benchmark:
        # warm ups
        for i in range(warm_ups):
            outputs = session.run_with_ort_values([output_name], input_dict)
        
        start = time.time()
        for i in range(repeats):
            outputs = session.run_with_ort_values([output_name], input_dict)
        end = time.time()
        iter_time = (end - start) / repeats * 1000
        logging.info(f"Elapsed Time: {iter_time}ms")
    else:
        start = time.time()
        outputs = session.run_with_ort_values([output_name], input_dict)
        end = time.time()
        iter_time = (end - start)* 1000
        logging.info(f"Elapsed Time: {iter_time}ms")

    return (outputs[0].numpy(), iter_time)


def run_ait(benchmark: bool, bs:int, hs: int):
    raise NotImplementedError("run_ait not implemented yet")


@click.command()
@click.option("--all", is_flag=True)
@click.option("--make_graph", is_flag=True)
@click.option("--build", is_flag=True)
@click.option("-r", "--run", default="")
@click.option("--benchmark", is_flag=True)
@click.option("-b", "--batch_size", default=batch_size_default)
@click.option("-h", "--hidden_size", default=hidden_size_default)
@click.option("-n", default=num_elementwise_default)
@click.option("--ait_path", help="location of the AIT sources (to include the headers during compilation) e.g., /work/AITemplate/", default="/work/AITemplate/")
@click.option("--onnx_path", help="location of onnx headers (e.g., /work/onnxruntime/include/)", default="/work/onnxruntime/include/")
def _run(all: bool, make_graph: bool, build: bool, run: str, benchmark: bool, batch_size: int, hidden_size: int, n: int, ait_path: str, onnx_path: str):
    logging.getLogger().setLevel(logging.INFO)
    if all or build or make_graph:
        create_graph(batch_size, hidden_size, n)

    if all or build:
        build_custom_op(ait_path, onnx_path)

    assert run in {"", "custom", "original", "ait_generated", "ait"}
    if run == "custom":
        run_onnx(custom=True, benchmark=benchmark, bs=batch_size, hs=hidden_size)
    elif run == "original":
        run_onnx(custom=False, benchmark=benchmark, bs=batch_size, hs=hidden_size)
    elif run == "ait_generated" or run == "ait":
        run_ait(benchmark=benchmark, bs=batch_size, hs=hidden_size)


if __name__ == "__main__":
    _run()