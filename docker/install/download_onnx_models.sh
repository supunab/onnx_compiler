#!/bin/bash

# download bert and gpt2 models (to make it eaier to reproduce examples) and put in /onnx_models/ folder
# bert (saved in /onnx_models/bert_base_cased_3_fp16_gpu.onnx)
python3 -m onnxruntime.transformers.benchmark -m bert-base-cased -b 1 -t 10 -f fusion.csv -r result.csv -d detail.csv -c ./cache_models --onnx_dir ./onnx_models -o by_script -g -p fp16 -i 3 --use_mask_index --overwrite

# distilgpt2 (saved in /onnx_models/distilgpt2_fp16.onnx)
python3 -m onnxruntime.transformers.models.gpt2.convert_to_onnx -m distilgpt2 --output /onnx_models/distilgpt2_fp16.onnx -o -p fp16 --use_gpu
