from common import *
import onnx
import sys
import os

sys.path.insert(1, os.path.abspath("./../../"))
from converter import transform_graph, compile, remove_attention_mask_hack
from custom_op_generator import generate, convert_graph
import logging
import click
from common import *

def build_gpt2(save_transform: bool, do_compile: bool, visualize_ait_graph: bool, debug_logs: bool, model_path: str, ait_build_folder: str, ait_path: str, onnx_path: str,
                batch_size: int, hidden_size: int, past_seq_len: int, curr_seq_len: int, vocab_size: int, attn_type: str):

    if debug_logs:
        logging.Logger.setLevel(logging.getLogger(), logging.DEBUG)

    if (not do_compile) and (not save_transform) and (not visualize_ait_graph):
        logging.info("You must specify either --save_transform (to save the transformed graph for " + \
                        " debug purposes) or --do_compile (to compile a custom op for the model)" + \
                        "--visualize_ait_graph (to visualize the optimized AIT graph)")

    assert attn_type=="mem_eff", "Only mem_eff attention is supported"

    # dim names should match the names in the onnx graph
    attributes = {
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "past_seq_len": past_seq_len,
        "seq_len": curr_seq_len,
        "total_seq_len": past_seq_len + curr_seq_len,
        "attn_type": "mem_eff"
    }

    model = onnx.load_model(model_path)

    converted_model_path = "distilgpt2_converted.onnx"
    model_name = "distilgpt2"

    if save_transform:
        transform_graph(model, attributes=attributes)
        remove_attention_mask_hack(model)
        onnx.save_model(model, "test.onnx")

    if do_compile:
        context = compile(model, output_dir=ait_build_folder, model_name=model_name, attributes=attributes)
        generate(context, os.path.join(ait_build_folder, model_name), ait_path=ait_path, onnx_header_path=onnx_path)
        convert_graph(model.graph, context, converted_model_path)

    if visualize_ait_graph:
        outputs = compile(model, output_dir=ait_build_folder, model_name=model_name, attributes=attributes, return_out=True)
        # pick the last output?
        # Y = outputs[-1]
        from aitemplate import compiler
        from aitemplate.testing import detect_target
        from aitemplate.utils.visualization import plot_graph
        def apply_optimizations(tensors):
            target = detect_target()
            # first, convert output tensors to graph
            with target:
                graph = compiler.transform.toposort(tensors)
                # second, provide names to the graph
                compiler.transform.name_graph(graph)
                compiler.transform.mark_param_tensor(graph)
                compiler.transform.mark_special_views(graph)
                # we can apply optimizations to the graph, or test single optimization pass on the graph
                graph = compiler.transform.optimize_graph(graph, "./tmp")
            return graph
        graph = apply_optimizations(outputs)
        plot_graph(graph, file_path="./ait_gpt2_model.html")


@click.command()
@click.option('-st', "--save_transform", is_flag=True)
@click.option('-c', "--do_compile", is_flag=True)
@click.option("--visualize_ait_graph", is_flag=True)
@click.option('-d', "--debug_logs", is_flag=True)
@click.option('-m', "--model_path", default="/onnx_models/distilgpt2_fp16.onnx", help="optimized ONNX model path for BERT")
@click.option('-ait', "--ait_build_folder", default="./tmp/", help="path to generate AIT sources")
@click.option("--ait_path", help="location of the AIT sources (to include the headers during compilation) e.g., /AITemplate/", default="/AITemplate/")
@click.option("--onnx_path", help="location of onnx headers (e.g., /onnxruntime/include/)", default="/onnxruntime/include/")
@click.option("--batch_size", default=batch_size_default)
@click.option("--hidden_size", default=hidden_size_default)
@click.option("--past_seq_len", default=past_seq_len_default)
@click.option("--curr_seq_len", default=curr_seq_len_default)
@click.option("--vocab_size", default=vocab_size_default)
@click.option("--attn_type", default="mem_eff", help="Choose one of {mem_eff, flash, unfused, default}")
def _build_gpt2(save_transform: bool, do_compile: bool, visualize_ait_graph: bool, debug_logs: bool, model_path: str, ait_build_folder: str, ait_path: str, onnx_path: str, 
         batch_size: int, hidden_size: int, past_seq_len: int, curr_seq_len: int, vocab_size: int, attn_type: str):
    build_gpt2(save_transform, do_compile, visualize_ait_graph, debug_logs, model_path, ait_build_folder, ait_path, onnx_path, batch_size, hidden_size, past_seq_len, curr_seq_len, vocab_size, attn_type)

if __name__ == "__main__":
    _build_gpt2()
