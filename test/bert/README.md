## BERT
If you don't have the ONNX graph for BERT, get that by running the transformers tool in ORT.
```bash
python -m onnxruntime.transformers.benchmark -m bert-base-cased -b 1 -t 10 -f fusion.csv -r result.csv -d detail.csv -c ./cache_models --onnx_dir ./onnx_models -o by_script -g -p fp16 -i 3 --use_mask_index --overwrite

# You can find an onnx model ./onnx_models/bert-base-cased_fp16.onnx
```

Then, we need to build the custom op. For that use `build_bert.py`. You should pass the paths to onnx model, onnx include files, etc. when running this. Run `python3 build_bert.py --help` to see the available command line args (e..g, batch size, sequence length, attention type, etc.).

```bash
python3 build_bert.py --do_compile --ait_path <path of ait sources> --onnx_path <path of onnx headers>
```

This should create a shared object for the compiled custom op (if default paths were used, this should be in `./tmp/test_model/test.so`) and the converted model (default `model_converted.onnx`)

Once you have the compiled shared op, to run it use the `run_bert.py` model. You can use `run_bert.py --help` to pick the version you want to run. For example, to benchmark
time it takes for the compiled custom op use `python3 run_bert.py --run_custom --benchmark`.