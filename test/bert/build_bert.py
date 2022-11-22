"""
Build a custom op that wraps the AIT generated code for BERT
------------------------------------------------------------
(Can find more details about args here: https://github.com/huggingface/transformers/blob/aa39967b2898d1056d51ec3b710468ca95773074/src/transformers/models/bert/modeling_bert.py#L812)
Inputs:
    input_ids: [batch_size, seq_length] - input token ids
    token_type_ids: [batch_size, seq_lenght] - to identify the segment tokens belong to. 0 for segment "A", 1 for segment "B"
                                               (used for cases where input_ids corresponds to two [or more?] sentences)
    attention_mask: [batch_size, seq_length] - masks padding as 0

Outputs:
    (embed_size = 768)
    output1: [batch_size, seq_length, embed_size] - embeddings of each token
    output2: take (gather) the first output token embedding of each sentence --> matmul + bias --> tanh 

**inputs and outputs are provided in float32, but all the ops are float16 ops. Therefore, casts are added for inputs and outputs
    to convert them back to/from float32
"""
import onnx
import sys
import os

sys.path.insert(1, os.path.abspath("./../../"))
from converter import transform_graph, compile
from custom_op_generator import generate, convert_graph
import logging
import click
from common import *

@click.command()
@click.option('-st', "--save_transform", is_flag=True)
@click.option('-c', "--do_compile", is_flag=True)
@click.option("--visualize_ait_graph", is_flag=True)
@click.option('-d', "--debug_logs", is_flag=True)
@click.option('-m', "--model_path", default="/work/models/bert_base/onnx_models/bert_base_cased_3_fp16_gpu.onnx", help="optimized ONNX model path for BERT")
@click.option('-ait', "--ait_build_folder", default="./tmp/", help="path to generate AIT sources")
@click.option("--ait_path", help="location of the AIT sources (to include the headers during compilation) e.g., /work/AITemplate/", default="/work/AITemplate/")
@click.option("--onnx_path", help="location of onnx headers (e.g., /work/onnxruntime/include/)", default="/work/onnxruntime/include/")
def _run(save_transform: bool, do_compile: bool, visualize_ait_graph: bool, debug_logs: bool, model_path: str, ait_build_folder: str, ait_path: str, onnx_path: str):
    """
    --to generate the optimized fp16 cuda onnx model for bert using ort--
    python -m onnxruntime.transformers.benchmark -m bert-base-cased -b 1 -t 10 -f fusion.csv -r result.csv -d detail.csv -c ./cache_models --onnx_dir ./onnx_models -o by_script -g -p fp16 -i 3 --use_mask_index --overwrite
    You can find an onnx model ./onnx_models/bert-base-cased_fp16.onnx
    """
    if debug_logs:
        logging.Logger.setLevel(logging.getLogger(), logging.DEBUG)

    if (not do_compile) and (not save_transform) and (not visualize_ait_graph):
        logging.info("You must specify either --save_transform (to save the transformed graph for " + \
                        " debug purposes) or --do_compile (to compile a custom op for the model)" + \
                        "--visualize_ait_graph (to visualize the optimized AIT graph)")
    
    # TODO: attention_mask is ignored at the moment since there's no matching param in AIT~ Bert
    converted_model_path = "bert_converted.onnx"
    model_name = "bert"

    attributes={
        "batch_size": batch_size,
        "hidden_size": hidden_size, 
        "seq_len": seq_len}

    model = onnx.load_model(model_path)

    if save_transform:
        transform_graph(model, attributes={"batch_size": batch_size, "hidden_size": hidden_size, "seq_len": seq_len})
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
        plot_graph(graph, file_path="ait_bert_model.html", network_name="ait_bert")

if __name__ == "__main__":
    _run()
