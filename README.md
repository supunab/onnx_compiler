## ONNX Compiler (via AITemplate)

This prototype uses AITemplate as a backend to generate code for ONNX graphs. Specifically, we use generated AITemplate code to build a custom op that can be loaded inside
ONNXRuntime (ORT). Then the original ONNX graph is converted to contain the custom op so that it can be executed in ORT. 

The current version compiles the entire ONNX graph into a single custom op, but ideally this should have the capability to pick subgraphs you need to optimize and to compile such subgraphs into custom ops using AITemplate.

### Use Docker (recommended)
We have created a docker image with all the dependencies and example ONNX models to make it easier to run experiments and examples. This is the recommended way of running.

To build the docker image, use the following command.

```
docker build -t onnx_compiler -f docker/Dockerfile.cuda .
```

To run (an interractive session `-it`)
```
docker run --gpus all --user user -it onnx_compiler:latest
```

**Important Note: ** Due to [this issue](https://github.com/supunab/onnx_compiler/issues/5), you have to manually run `docker/install/download_onnx_models.sh` script to download the ONNX models for BERT and GPT2 and save it in the default directory. Specifically, use the following command when you first start your docker image.

```
bash /onnx_compiler/docker/install/download_onnx_models.sh
```

To start a previously stopped image, fisrt make sure you have the image by running `docker container ls -a`. Then find the `container_id`. Then you can simply start the container
using `docker exec -it <container_id>`.



### Implementation
Most of the implementation is in the main directory of the repo. `converter.py` has the `compile` method that compiles the ONNX graph using AITemplate. This uses `registery.py` to map ONNX ops into AITemplate ops. In certain cases, we have to transform the original ONNX graph to make this conversion convenient (e.g., unfuse nodes, fuse nodes, rewrite nodes, etc.). Once the compilation is done, we can generate the code for creating an ORT custom op using `generate` method in `custom_op_generator.py`. This will generate and compile a shared-object that can be loaded to ORT to use the custom op. Finally, we need to convert the original ONNX graph to a graph that contains the newly compiled custom op. For that we can use the `convert_graph` method in `custom_op_generator.py`.

### Examples
You find several examples in the `examples` directory. 

### Requirements
You need to install ONNXRuntime and AITemplate 

Versions (commit hashes) tested are below:
AITemplate (my fork) - https://github.com/supunab/AITemplate/commit/aee58a136d663a855445d45565ce13daf3b7e8d2
ONNXRuntime - 1.13 release
ONNXRuntime headers are cloned from - https://github.com/microsoft/onnxruntime/tree/b353e0b41d588605958b03f9a223d10a2fbeb514

