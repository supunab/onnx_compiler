"""
Benchmark BERT for different batch sizes and sequence lengths
"""
from run_bert import run_bert, original_model_path
from build_bert import build_bert
from common import *
import time
import logging
import subprocess
import sys
import re
pattern = re.compile("Elapsed Time: [0-9]+\.[0-9]*ms")

def write_result(res_name: str, iter_time: float, fname: str):
    with open(fname, "a") as f:
        f.write(f"{res_name},{iter_time}\n")

def parse_time(text: str):
    t = pattern.findall(text)
    if len(t) == 0:
        return -1
    else:
        assert len(t) == 1, "Should only have one elapsed time value"
        return float(t[0].split(":")[1][1:-2])

def append_log(fname: str, text: str):
    with open(fname, "a") as f:
        f.write(text + "\n\n\n")

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    seq_lens = [64, 128, 384, 512, 1024, 2048, 4096]
    hidden_sizes = [hidden_size_default]
    attn_type = ["flash", "mem_eff", "unfused"]

    # some common params
    ait_build_folder = "./tmp/"
    ait_path = "/work/supun/AITemplate/"
    onnx_path = "/work/supun/onnxruntime/include/"

    data = {} # store the results
    fname = f"results_{str(time.time()).split('.')[0]}.csv"
    log_file = f"log_{str(time.time()).split('.')[0]}.log"
    for bs in batch_sizes:
        for sl in seq_lens:
            for hs in hidden_sizes:
                # run onnx original
                case_name = f"onnx_original_bert_bs_{bs}_hs{hs}_sl{sl}"
                logging.info(f"Running {case_name}")
                out = subprocess.run(f"python3 run_bert.py --run_original --benchmark --batch_size {bs} --hidden_size {hs} --seq_len {sl} --vocab_size {vocab_size_default}", shell=True, capture_output=True)
                append_log(log_file, out.stderr.decode())
                iter_time = parse_time(out.stderr.decode())
                data[case_name] = iter_time
                write_result(f"onnx_original,{bs},{hs},{sl}", iter_time, fname)
                for at in attn_type:
                    # run AIT custom op
                    case_name = f"onnx_ait_custom_op_bert_bs_{bs}_hs{hs}_sl{sl}_attn_{at}"
                    logging.info(f"Running {case_name}")
                    # build custom op
                    subprocess.run(f"python3 build_bert.py --do_compile -m {original_model_path} -ait {ait_build_folder} --ait_path {ait_path} --onnx_path {onnx_path} --batch_size {bs} --hidden_size {hs} --seq_len {sl} --vocab_size {vocab_size_default} --attn_type {at}", shell=True)
                    
                    # run
                    out = subprocess.run(f"python3 run_bert.py --run_custom --benchmark --batch_size {bs} --hidden_size {hs} --seq_len {sl} --vocab_size {vocab_size_default}", shell=True, capture_output=True)
                    append_log(log_file, out.stderr.decode())
                    iter_time = parse_time(out.stderr.decode())
                    data[case_name] = iter_time
                    write_result(f"ORT (AIT Custom op: attn_{at}),{bs},{hs},{sl}", iter_time, fname)

                    # run AIT directly
                    case_name = f"ait_bert_bs_{bs}_hs{hs}_sl{sl}_attn_{at}"
                    logging.info(f"Running {case_name}")
                    out = subprocess.run(f"python3 run_bert.py --run_ait_generated --benchmark --batch_size {bs} --hidden_size {hs} --seq_len {sl} --vocab_size {vocab_size_default}", shell=True, capture_output=True)
                    append_log(log_file, out.stderr.decode())
                    iter_time = parse_time(out.stderr.decode())
                    data[case_name] = iter_time
                    write_result(f"AIT: attn_{at},{bs},{hs},{sl}", iter_time, fname)

