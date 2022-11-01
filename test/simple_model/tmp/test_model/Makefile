
CC = nvcc
CFLAGS = 
fPIC_flag = -Xcompiler=-fPIC

obj_files = gemm_rcr_bias_relu_8.obj gemm_rcr_bias_relu_9.obj gemm_rcr_bias_relu_10.obj gemm_rcr_bias_6.obj softmax_7.obj model_container_base.obj utility.obj model_container.obj model_interface.obj ort_ait_custom_op_library.obj

%.obj : %.cu
	nvcc -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_USE_TANH_FOR_SIGMOID=1 -w -gencode=arch=compute_80,code=[sm_80,compute_80] -Xcompiler=-fPIC -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -Xcompiler -fvisibility=hidden -O3 -std=c++17 --expt-relaxed-constexpr --use_fast_math -I/work/AITemplate/3rdparty/cutlass/include -I/work/AITemplate/3rdparty/cutlass/tools/util/include -I/work/AITemplate/3rdparty/cutlass/examples/35_gemm_softmax -I/work/AITemplate/3rdparty/cutlass/../../python/aitemplate/backend/cuda/attention/src/./ -I/work/AITemplate/3rdparty/cutlass/../../python/aitemplate/backend/cuda/attention/src/fmha -I/work/onnxruntime/include/ -I/work/onnxruntime/include//onnxruntime/core/session/ -c -o $@ $<
%.obj : %.bin
	ld -r -b binary -o $@ $<

.PHONY: all
all: test.so

test.so: $(obj_files)
	$(CC) -shared $(fPIC_flag) $(CFLAGS) -o $@ $(obj_files)

clean:
	rm -f $(obj_files) test.so