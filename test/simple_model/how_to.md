### How to run
1. First, generate the ONNX model by running `python3 simple_onnx_model.py`
2. Then, generate AIT sources and generate a wrapper ORT custom op using `python3 build_custom_op.py`
3. Can now run all three versions using `python3 test_compiled_custom_op.py`