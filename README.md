## ONNX Compiler (via AITemplate)

This prototype uses AITemplate as a backend to generate code for ONNX graphs. Specifically, we use generated AITemplate code to build a custom op that can be loaded inside
ONNXRuntime (ORT). Then the original ONNX graph is converted to contain the custom op so that it can be executed in ORT. 

The current version compiles the entire ONNX graph into a single custom op, but ideally this should have the capability to pick subgraphs you need to optimize and to compile such subgraphs into custom ops using AITemplate.

### Implementation
Most of the implementation is in the main directory of the repo. `converter.py` has the `compile` method that compiles the ONNX graph using AITemplate. This uses `registery.py` to map ONNX ops into AITemplate ops. In certain cases, we have to transform the original ONNX graph to make this conversion convenient (e.g., unfuse nodes, fuse nodes, rewrite nodes, etc.). Once the compilation is done, we can generate the code for creating an ORT custom op using `generate` method in `custom_op_generator.py`. This will generate and compile a shared-object that can be loaded to ORT to use the custom op. Finally, we need to convert the original ONNX graph to a graph that contains the newly compiled custom op. For that we can use the `convert_graph` method in `custom_op_generator.py`.

### Examples
You find several examples in the `test` directory. 

### Requirements
You need to install ONNXRuntime and AITemplate 