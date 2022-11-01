"""
Load the compiled custom op and verify it produces the correct result by checking
the output against PyTorch

TODO(supuna): I still get the following error if I use onnxruntime directly from pip. But it works fine in the
              docker container that has ort-cuda built from sources. This could be because CUDA EP is not properly
              configured and the fallback CPU EP doesn't have support for FP16.

onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph: [ONNXRuntimeError] : 10 : INVALID_GRAPH : This is an invalid model. Type Error: Type 'tensor(float)' ....
"""
import onnxruntime as ort
import numpy as np
from simple_onnx_model import SimpleModel, test_pytorch
import torch.utils.benchmark as benchmark

def _benchmark_onnx(session, io_binding):
    session.run_with_iobinding(io_binding)

def run_onnx_model(model_path: str, input_np: np.ndarray, shared_lib: bool = False, shared_lib_path: str = ""):
    from constants import input_size, num_layers, hidden_sizes, warm_ups, repeats
    session_options = ort.SessionOptions()
    if shared_lib:
        session_options.register_custom_ops_library(shared_lib_path)
    session = ort.InferenceSession(model_path, sess_options=session_options, providers=["CUDAExecutionProvider"])

    input_ort = ort.OrtValue.ortvalue_from_numpy(input_np, "cuda", 0)

    output_ort = ort.OrtValue.ortvalue_from_shape_and_type((batch_size, hidden_sizes[-1]), np.float16, "cuda", 0)
    # note - weights for initializers is already in the onnx graph

    io_binding = session.io_binding()
    input_name = session.get_inputs()[0].name
    io_binding.bind_input(name=input_name, device_type=input_ort.device_name(), device_id=0, element_type=np.float16,
                             shape=input_ort.shape(), buffer_ptr=input_ort.data_ptr())
    
    output_name = session.get_outputs()[0].name
    io_binding.bind_output(name=output_name, device_type=output_ort.device_name(), device_id=0, 
                        element_type=np.float16, shape=output_ort.shape(), buffer_ptr=output_ort.data_ptr())

    timer = benchmark.Timer(
        stmt="_benchmark_onnx(session, io_binding)",
        setup="from __main__ import _benchmark_onnx",
        globals={"session": session, "io_binding" : io_binding}
    )
    timer.timeit(warm_ups) # warm up
    print(f"""{"ORT (AIT Custom Op)" if shared_lib else "ORT Original"}: {timer.timeit(repeats)}""")

    # session.run_with_iobinding(io_binding)
    return output_ort.numpy()

if __name__ == "__main__":
    shared_library = "./tmp/test_model/test.so"
    model_path = "./simple_converted.onnx"
    original_model_path = "./simple.onnx"

    from constants import input_size, batch_size
    # use same input for all variants
    input_np = np.random.rand(batch_size, input_size).astype(np.float16)

    ort_output = run_onnx_model(original_model_path, input_np)
    ait_custom_op_output = run_onnx_model(model_path, input_np, True, shared_library)
    pt_output = test_pytorch(input_np)

    # print(ort_output)
    # print(ait_custom_op_output)
    # print(pt_output)
    equal = np.allclose(ait_custom_op_output, pt_output, atol=1e-1) and  np.allclose(ait_custom_op_output, ort_output, atol=1e-1)
    if equal:
        print("Outputs matched! (to a 0.1 tolerenace)")
    else:
        print("ERROR! outputs are different!")