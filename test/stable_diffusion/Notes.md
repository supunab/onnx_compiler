## Notes

This sort of hit an unexpected road block. Right now, what I'm doing is importing the pretrained SD UNet model from HuggingFace, exporting it using the torch
ONNX exporter, separately use AITemplate given example to generate the AIT compiled code for SD UNet and then, generating an interface to that generated files
using the exported ONNX graph.

I use the same naming mechanism as Torch/AIT to make sure I load the same pretrained weights and to make sure I pass the values correctly from ORT inputs/initializers
to AIT interface to run the model. This works if the "atomic" operators in both cases remain the same so that the ORT inputs/initializers have a one-to-one mapping to
the corresponding AIT weights. However, this doesn't seem to be the case (and I should've seen this coming!). Torch onnx exporter doesn't gurantee that we will keep
using the same set of ops in the onnx graph and it is infact performs operations in a different way. For example (an artificial example), a single matmul using a single
weight tensor in AIT can be converted into two gemms with two separate weights. Now, there's no way of recovering the mapping from this two weights to the original single
weight, hence, we cannot do the ORT weights --> AIT weight mapping for such cases.


### Current Workflow
1. Use `my_test.py` in `AITemplate/examples/05_stable_diffusion/` to generate the AIT sources for SD UNet model
    - This will save the `sd_unet.onnx` in the specified directory
2. Use `stable_diffusion_build.py` in this directory to generate a custom op header for ORT that interaces with the generated AIT code
3. Move the generated `ort_ait_custom_op_library.cu` to the place where the previously generated AIT sources lie
4. Use `make` to compile and generate the `test.so`
5. Then use `stable_diffusion_run.py` to run the model
- unfortunately, due to reasons specified in the above section, this won't work :( 