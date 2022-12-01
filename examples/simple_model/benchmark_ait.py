import sys
import os

sys.path.insert(1, os.path.abspath("./../../"))

from pyexpat import model
from aitemplate.compiler import Model
import onnx
from utils import map_onnx_dtype_to_numpy, clean_name
from simple_onnx_model import test_pytorch
import numpy as np
import torch
import torch.utils.benchmark as benchmark

from constants import *

def _benchmark_ait(mod, input_pt, output_pt):
    # sync = True to measure true time
    mod.run_with_tensors([input_pt], [output_pt], sync = True)

if __name__ == "__main__":
    # load the compiled test.so
    model_path = "./tmp/test_model/"
    mod = Model(os.path.join(model_path, "test.so"))

    # load onnx graph and map initializers weights to AIT constants
    onnx_graph_path = "simple.onnx"
    onnx_model = onnx.load_model(onnx_graph_path)
    graph = onnx_model.graph
    for init in graph.initializer:
        name = clean_name(init.name)
        dtype = map_onnx_dtype_to_numpy(init.data_type)
        shape = list(init.dims)
        np_data = np.frombuffer(init.raw_data, dtype=dtype).reshape(shape).copy() # copy to make it writable (to make PyTorch happy)
        tensor = torch.from_numpy(np_data).cuda()
        mod.set_constant_with_tensor(name, tensor)
    
    # run the model with input
    input_np = np.random.rand(batch_size, input_size).astype(np.float16)
    input_pt = torch.from_numpy(input_np).cuda()

    output_pt = torch.empty(batch_size, hidden_sizes[-1]).cuda().half()
    # mod.run_with_tensors([input_pt], [output_pt])

    timer = benchmark.Timer(
        stmt="mod.run_with_tensors([input_pt], [output_pt])",
        setup="from __main__ import _benchmark_ait",
        globals={"mod": mod, "input_pt": input_pt, "output_pt": output_pt}
    )

    timer.timeit(warm_ups) # warmp ups
    print(f"AIT Time: {timer.timeit(repeats)}")

    # cross check result with PyTorch
    pt_output = test_pytorch(input_np)
    ait_output = output_pt.detach().cpu().numpy()
    equal = np.allclose(ait_output, pt_output, atol=1e-1)
    if equal:
        print("Outputs matched! (to a 0.1 tolerenace)")
    else:
        print("ERROR! outputs are different!")
        