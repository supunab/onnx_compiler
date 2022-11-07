"""
Load the compiled model from test.so and run it directly in AIT
"""
from aitemplate.compiler import Model
from aitemplate.frontend import Tensor
import numpy as np
from common import *
import torch

def map_unet_params(pt_mod, dim, so: Model):
    pt_params = dict(pt_mod.named_parameters())
    del pt_mod
    params_ait = {}
    for key, arr in pt_params.items():
        if len(arr.shape) == 4:
            temp = arr.permute((0, 2, 3, 1)).contiguous()
            del arr
            so.set_constant_with_tensor(key.replace(".", "_"), temp)
            continue
        elif key.endswith("ff.net.0.proj.weight"):
            w1, w2 = arr.chunk(2, dim=0)
            so.set_constant_with_tensor(key.replace(".", "_"), w1)
            so.set_constant_with_tensor(key.replace(".", "_").replace("proj", "gate"), w2)
            continue
        elif key.endswith("ff.net.0.proj.bias"):
            w1, w2 = arr.chunk(2, dim=0)
            so.set_constant_with_tensor(key.replace(".", "_"), w1)
            so.set_constant_with_tensor(key.replace(".", "_").replace("proj", "gate"), w2)
            continue
        so.set_constant_with_tensor(key.replace(".", "_"), arr)

    arange_tensor = torch.arange(start=0, end=dim // 2, dtype=torch.float32).cuda().half()
    so.set_constant_with_tensor("arange", arange_tensor)

def benchmark_ait():
    so_path = "./ait_generated_sources/test.so"
    so = Model(so_path)
    
    # set AIT weights from hf model
    pt_model = get_pt_sd_unet_from_hf()
    map_unet_params(pt_model, dim, so)

    # inputs
    np.random.seed(0)
    latent_model_input = np.random.rand(batch_size, input_channels, hh, ww).astype(np.float16)
    timesteps = np.random.rand(batch_size).astype(np.float16)
    text_embeddings = np.random.rand(batch_size, prompt_length, embedding_size).astype(np.float16)

    latent_model_input_pt = torch.from_numpy(latent_model_input).cuda()
    timesteps_pt = torch.from_numpy(timesteps).cuda()
    text_embeddings_pt = torch.from_numpy(text_embeddings).cuda()

    inputs = {
        "input0": latent_model_input_pt.permute((0, 2, 3, 1)).contiguous(),
        "input1": timesteps_pt,
        "input2": text_embeddings_pt,
    }

    ys = []
    num_ouputs = len(so.get_output_name_to_index_map())
    for i in range(num_ouputs):
        shape = so.get_output_maximum_shape(i)
        ys.append(torch.empty(shape).cuda().half())
    so.run_with_tensors(inputs, ys)

if __name__ == "__main__":
    benchmark_ait()