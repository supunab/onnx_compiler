"""
Load the compiled custom op and verify it produces the correct result by checking
the output against PyTorch

TODO(supuna): I still get the following error if I use onnxruntime directly from pip. But it works fine in the
              docker container that has ort-cuda built from sources. This could be because CUDA EP is not properly
              configured and the fallback CPU EP doesn't have support for FP16.

onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph: [ONNXRuntimeError] : 10 : INVALID_GRAPH : This is an invalid model. Type Error: Type 'tensor(float)' ....
"""
import onnxruntime as ort
import onnx
import numpy as np
import torch
from simple_onnx_model import SimpleModel, test_pytorch

if __name__ == "__main__":
    shared_library = "./../tmp/test_model/test.so"
    model_path = "./../simple_converted.onnx"

    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(shared_library)
    onnx_model = onnx.load_model(model_path)
    session = ort.InferenceSession(model_path, sess_options=session_options, providers=["CUDAExecutionProvider"])

    # setup data
    input_size = 10
    hidden_size = 25
    output_size = 5
    batch_size = 8

    input_np = np.random.rand(batch_size, input_size).astype(np.float16)
    input_ort = ort.OrtValue.ortvalue_from_numpy(input_np, "cuda", 0)

    output_ort = ort.OrtValue.ortvalue_from_shape_and_type((batch_size, output_size), np.float16, "cuda", 0)

    # TODO: weights already stored in the onnx model should work as well?
    # initializers (weights)
    # dense1weight = np.random.rand(input_size, hidden_size).astype(np.float16)
    # dense1bias = np.random.rand(hidden_size).astype(np.float16)
    # dense2weight = np.random.rand(input_size, hidden_size).astype(np.float16)
    # dense2bias = np.random.rand(hidden_size).astype(np.float16)

    io_binding = session.io_binding()
    input_name = session.get_inputs()[0].name
    io_binding.bind_input(name=input_name, device_type=input_ort.device_name(), device_id=0, element_type=np.float16,
                             shape=input_ort.shape(), buffer_ptr=input_ort.data_ptr())
    
    output_name = session.get_outputs()[0].name
    io_binding.bind_output(name=output_name, device_type=output_ort.device_name(), device_id=0, 
                        element_type=np.float16, shape=output_ort.shape(), buffer_ptr=output_ort.data_ptr())

    session.run_with_iobinding(io_binding)
    ait_output = output_ort.numpy()
    pt_output = test_pytorch(input_size, hidden_size, output_size, input_np)
    print(ait_output)
    print(pt_output)

