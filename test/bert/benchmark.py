"""
Benchmark BERT for different batch sizes and sequence lengths
"""
from run_bert import run_bert, original_model_path
from build_bert import build_bert
from common import *
import time
import logging

def write_result(res_name: str, iter_time: float, fname: str):
    with open(fname, "a") as f:
        f.write(f"{res_name},{iter_time}\n")

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    # batch_sizes = [batch_size_default]
    seq_lens = [64, 128, 384, 512]
    # seq_lens = [seq_len_default]
    hidden_sizes = [hidden_size_default]
    attn_type = ["flash", "mem_eff", "unfused"]

    # some common params
    ait_build_folder = "./tmp/"
    ait_path = "/work/supun/AITemplate/"
    onnx_path = "/work/supun/onnxruntime/include/"

    data = {} # store the results
    fname = f"results_{str(time.time()).split('.')[0]}"
    for bs in batch_sizes:
        for sl in seq_lens:
            for hs in hidden_sizes:
                # run onnx original
                case_name = f"onnx_original_bert_bs_{bs}_hs{hs}_sl{sl}"
                logging.info(f"Running {case_name}")
                iter_time = run_bert(True, False, False, True, bs, hs, sl, vocab_size_default)
                data[case_name] = iter_time
                write_result(f"onnx_original,{bs},{hs},{sl}", iter_time, fname)
                for at in attn_type:
                    # run AIT custom op
                    case_name = f"onnx_ait_custom_op_bert_bs_{bs}_hs{hs}_sl{sl}_attn_{at}"
                    logging.info(f"Running {case_name}")
                    build_bert(False, True, False, False, original_model_path, ait_build_folder, ait_path, onnx_path, bs, hs, sl, vocab_size_default, at)
                    iter_time = run_bert(False, True, False, True, bs, hs, sl, vocab_size_default)
                    data[case_name] = iter_time
                    write_result(f"ORT (AIT Custom op: attn_{at}),{bs},{hs},{sl}", iter_time, fname)


                    # run AIT directly
                    case_name = f"ait_bert_bs_{bs}_hs{hs}_sl{sl}_attn_{at}"
                    logging.info(f"Running {case_name}")
                    iter_time = run_bert(False, False, True, True, bs, hs, sl, vocab_size_default)
                    data[case_name] = iter_time
                    write_result(f"AIT: attn_{at},{bs},{hs},{sl}", iter_time, fname)


    


