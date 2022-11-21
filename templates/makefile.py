import jinja2

# onnx_header_path=/work/onnxruntime/include/
# ait_source_path=/work/AITemplate

MAKEFILE_TEMPLATE = jinja2.Template("""
CC = nvcc
CFLAGS = 
fPIC_flag = -Xcompiler=-fPIC

onnx_header_path={{onnx_header_path}}
ait_source_path={{ait_path}}

obj_files = {{obj_files}}

%.obj : %.cu
	nvcc -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_USE_TANH_FOR_SIGMOID=1 -w -gencode=arch=compute_80,code=[sm_80,compute_80] -Xcompiler=-fPIC -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -Xcompiler -fvisibility=hidden -O3 -std=c++17 --expt-relaxed-constexpr --use_fast_math -I$(ait_source_path)/3rdparty/cutlass/include -I$(ait_source_path)/3rdparty/cutlass/tools/util/include -I$(ait_source_path)/3rdparty/cutlass/examples/35_gemm_softmax -I$(ait_source_path)/3rdparty/cutlass/../../python/aitemplate/backend/cuda/attention/src/./ -I$(ait_source_path)/3rdparty/cutlass/../../python/aitemplate/backend/cuda/attention/src/fmha -I$(onnx_header_path) -I$(onnx_header_path)/onnxruntime/core/session/ -c -o $@ $<
%.obj : %.bin
	ld -r -b binary -o $@ $<

.PHONY: all
all: test.so

test.so: $(obj_files)
	$(CC) -shared $(fPIC_flag) $(CFLAGS) -o $@ $(obj_files)

clean:
	rm -f $(obj_files) test.so
""")

