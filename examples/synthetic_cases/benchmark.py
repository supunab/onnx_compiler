"""
Benchmark n-elementwise for different n (back-to-back elementwise count)
"""
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
    n_ops = [1, 5, 10, 25, 50, 100]

    # some common params
    ait_path = "/AITemplate/"
    onnx_path = "/onnxruntime/include/"

    batch_size = 128
    hidden_size = 262144

    data = {} # store the results
    fname = f"results_{str(time.time()).split('.')[0]}.csv"
    log_file = f"log_{str(time.time()).split('.')[0]}.log"

    for n in n_ops:
        case_name = f"onnx_original_{n}"
        logging.info(f"Running {case_name}")
        # make the onnx graph for the given n
        subprocess.run(f"python3 synthetic_cases.py --make_graph --batch_size {batch_size} --hidden_size {hidden_size} -n {n}", shell=True, capture_output=True)
        # run
        out = subprocess.run(f"python3 synthetic_cases.py --run original --benchmark --batch_size {batch_size} --hidden_size {hidden_size}", shell=True, capture_output=True)
        append_log(log_file, out.stderr.decode())
        iter_time = parse_time(out.stderr.decode())
        data[case_name] = iter_time
        write_result(f"onnx_original,{n}", iter_time, fname)

        case_name = f"custom_op_{n}"
        logging.info(f"Running {case_name}")
        # build ait custom op
        subprocess.run(f"python3 synthetic_cases.py --build --batch_size {batch_size} --hidden_size {hidden_size} -n {n} --ait_path {ait_path} --onnx_path {onnx_path}", shell=True, capture_output=True)
        out = subprocess.run(f"python3 synthetic_cases.py --run custom --benchmark --batch_size {batch_size} --hidden_size {hidden_size}", shell=True, capture_output=True)
        append_log(log_file, out.stderr.decode())
        iter_time = parse_time(out.stderr.decode())
        data[case_name] = iter_time
        write_result(f"custom_op,{n}", iter_time, fname)
