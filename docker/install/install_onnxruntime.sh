#!/bin/bash

pip3 install onnxruntime-gpu

# need onnxruntime
# cloning the repo (TODO: don't need the entire repo but just the header folder)
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime
# checkout to release v1.13.1
git checkout tags/v1.13.1
