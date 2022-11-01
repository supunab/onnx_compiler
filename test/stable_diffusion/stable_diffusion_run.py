"""
Load the compiled shared object and run stable diffusion
"""
import onnxruntime as ort
from constants import *
import numpy as np

if __name__ == "__main__":
    model_path = "/work/models/sd_unet_converted.onnx"
    shared_library = "/work/AITemplate/examples/05_stable_diffusion/tmp/UNet2DConditionModel/test.so"
    sess_options = ort.SessionOptions()
    sess_options.register_custom_ops_library(shared_library)

    session = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CUDAExecutionProvider"])
    
    io_binding = session.io_binding()

    # setup inputs
    # (below are in order of the onnx graph)
    latent_model_input = np.random.rand(batch_size, hh, ww, input_channels).astype(np.float16)
    timesteps = np.random.rand(batch_size)
    text_embeddings = np.random.rand(batch_size, prompt_length, embedding_size).astype(np.float16)

    # ort_values
    latent_model_input_ort = ort.OrtValue.ortvalue_from_numpy(latent_model_input, "cuda", 0)
    timesteps_ort = ort.OrtValue.ortvalue_from_numpy(timesteps, "cuda", 0)
    text_embeddings_ort = ort.OrtValue.ortvalue_from_numpy(text_embeddings, "cuda", 0)
    
    # TODO: verify the output shapes
    output_ort = ort.OrtValue.ortvalue_from_shape_and_type([batch_size, hh, ww, input_channels], np.float16, "cuda", 0)

    input1_name = session.get_inputs()[0]
    input2_name = session.get_inputs()[1]
    input3_name = session.get_inputs()[2]

    io_binding.bind_input(name=input1_name, device_type=latent_model_input_ort.device_name(), device_id=0, element_type=np.float16,
                            shape=latent_model_input_ort.shape(), buffer_ptr=latent_model_input_ort.data_ptr())

    io_binding.bind_input(name=input2_name, device_type=timesteps_ort.device_name(), device_id=0, element_type=np.float16,
                            shape=timesteps_ort.shape(), buffer_ptr=timesteps_ort.data_ptr())

    io_binding.bind_input(name=input3_name, device_type=text_embeddings_ort.device_name(), device_id=0, element_type=np.float16,
                            shape=text_embeddings_ort.shape(), buffer_ptr=text_embeddings_ort.data_ptr())

    output_name = session.get_outputs[0]
    io_binding.bind_output(name=output_name, device_type=output_ort.device_name(), device_id=0, 
                        element_type=np.float16, shape=output_ort.shape(), buffer_ptr=output_ort.data_ptr())

    session.run_with_iobinding(io_binding)
    # print(output_ort.numpy())