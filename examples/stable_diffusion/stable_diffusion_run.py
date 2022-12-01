"""
Load the compiled shared object and run stable diffusion
"""
import onnxruntime as ort
from common import *
import numpy as np
import click


def run_pytorch(latent_model_input: np.ndarray, timesteps: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
    import torch
    model = get_pt_sd_unet_from_hf()
    latent_model_input = torch.from_numpy(latent_model_input).cuda()
    timesteps = torch.from_numpy(timesteps).cuda()
    text_embeddings = torch.from_numpy(text_embeddings).cuda()
    output = model(latent_model_input, timesteps, encoder_hidden_states=text_embeddings)
    return output.sample.detach().cpu().numpy()

def run_ort_custom_op(latent_model_input: np.ndarray, timesteps: np.ndarray, text_embeddings: np.ndarray, model_path: str, shared_lib: str) -> np.ndarray:
    sess_options = ort.SessionOptions()
    sess_options.register_custom_ops_library(shared_lib)
    session = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CUDAExecutionProvider"])

    io_binding = session.io_binding()

    # ort_values
    latent_model_input = latent_model_input.transpose(0, 2, 3, 1) # ait op requires them to be permuted (https://github.com/facebookincubator/AITemplate/blob/784206fde23f9f5153d328891125457c27739186/examples/05_stable_diffusion/benchmark.py#L92)
    latent_model_input_ort = ort.OrtValue.ortvalue_from_numpy(latent_model_input, "cuda", 0)
    timesteps_ort = ort.OrtValue.ortvalue_from_numpy(timesteps, "cuda", 0)
    text_embeddings_ort = ort.OrtValue.ortvalue_from_numpy(text_embeddings, "cuda", 0)

    output_ort = ort.OrtValue.ortvalue_from_shape_and_type([batch_size, hh, ww, input_channels], np.float16, "cuda", 0)

    input1_name = session.get_inputs()[0].name
    input2_name = session.get_inputs()[1].name
    input3_name = session.get_inputs()[2].name

    io_binding.bind_input(name=input1_name, device_type=latent_model_input_ort.device_name(), device_id=0, element_type=np.float16,
                            shape=latent_model_input_ort.shape(), buffer_ptr=latent_model_input_ort.data_ptr())

    io_binding.bind_input(name=input2_name, device_type=timesteps_ort.device_name(), device_id=0, element_type=np.float16,
                            shape=timesteps_ort.shape(), buffer_ptr=timesteps_ort.data_ptr())

    io_binding.bind_input(name=input3_name, device_type=text_embeddings_ort.device_name(), device_id=0, element_type=np.float16,
                            shape=text_embeddings_ort.shape(), buffer_ptr=text_embeddings_ort.data_ptr())

    output_name = session.get_outputs()[0].name
    io_binding.bind_output(name=output_name, device_type=output_ort.device_name(), device_id=0, 
                        element_type=np.float16, shape=output_ort.shape(), buffer_ptr=output_ort.data_ptr())
    session.run_with_iobinding(io_binding)

    # convert the output to original order
    return output_ort.numpy().transpose(0, 3, 1, 2)


@click.command()
@click.option("--so_path", default="./ait_generated_sources/test.so", help="Path to the compiled custom op shared object (.so)")
@click.option("--model_path", default="./sd_unet_converted.onnx", help="")
@click.option("--check_results", is_flag=True, help="Check the result against PyTorch output")
@click.option("--benchmark", is_flag=True, help="Benchmark against PyTorch")
def _run(so_path: str, model_path: str, check_results: bool, benchmark: bool):
    # setup inputs
    np.random.seed(0)
    latent_model_input = np.random.rand(batch_size, input_channels, hh, ww).astype(np.float16)
    timesteps = np.random.rand(batch_size).astype(np.float16)
    text_embeddings = np.random.rand(batch_size, prompt_length, embedding_size).astype(np.float16)

    print(latent_model_input)

    ait_custom_op_output = run_ort_custom_op(latent_model_input, timesteps, text_embeddings, model_path, so_path)
    if check_results:
        pt_output = run_pytorch(latent_model_input, timesteps, text_embeddings) # Note - this will OOM in my laptop GPU
        equal = np.allclose(ait_custom_op_output, pt_output, atol=1e-1)
        if equal:
            print("Outputs matched! (to a 0.1 tolerenace)")
        else:
            print("ERROR! outputs are different!")

    if benchmark:
        raise NotImplementedError("TODO: implement benchmarking against PyTorch")
    print(ait_custom_op_output)
    print(ait_custom_op_output.shape)

    # TODO: validate results!


if __name__ == "__main__":
    _run()