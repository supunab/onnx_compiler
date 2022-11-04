

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/util/host_tensor.h"


namespace {
#define GPU_1D_KERNEL_LOOP(i, n)   for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)


template <typename T, typename Telement, int element_in_Tio>
__global__ void nearest_upsampling_f16_nhwc_kernel(const T* input,
                                                    
                                                    T* output,
                                                    const int64_t batch,
                                                    const int64_t in_height,
                                                    const int64_t in_width,
                                                    const int64_t channels,
                                                    const int64_t out_height,
                                                    const int64_t out_width) {

    const float height_scale = in_height / static_cast<float>(out_height);
    const float width_scale = in_width / static_cast<float>(out_width);
    const int64_t nthreads = out_height * out_width * channels * batch;

GPU_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int out_x = n % out_width;
    n /= out_width;
    int out_y = n % out_height;
    n /= out_height;

    const T* bottom_data_n = input + n * channels * in_height * in_width;
    const int in_y =
        max(min(static_cast<int>(
                    floorf((static_cast<float>(out_y) + 0.5f) * height_scale)),
                static_cast<int>(in_height) - 1),
            0);
    const int in_x =
        max(min(static_cast<int>(
                    floorf((static_cast<float>(out_x) + 0.5f) * width_scale)),
                static_cast<int>(in_width) - 1),
            0);
    const int idx = (in_y * in_width + in_x) * channels + c;


  
    output[index] = __ldg(bottom_data_n + idx);
  

  }
}



template <typename integer>
constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

void bilinear_upsampling_luncher(cutlass::half_t* input,
                    
                      cutlass::half_t* output,
                      const int64_t N,
                      const int64_t H,
                      const int64_t W,
                      const int64_t C,
                      const int64_t HO,
                      const int64_t WO,
                      cudaStream_t stream) {
    const int64_t output_size = N * (C) * HO * WO;
    dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
    dim3 block(512);


    
    nearest_upsampling_f16_nhwc_kernel<float4, half, 8><<<grid, block, 0, stream>>>(
      (const float4 *)input,
      
      (float4 *)output,
      N, H, W, C/8, HO, WO);
    

}
} // namespace

void upsampling2d_540 (
    cutlass::half_t* in_ptr,
    
    cutlass::half_t* out_ptr,
    int64_t* batch,
    int64_t* in_h,
    int64_t* in_w,
    int64_t* in_ch,
    int64_t* out_batch,
    int64_t* out_h,
    int64_t* out_w,
    cudaStream_t stream
) {
  
  int64_t NI = *batch;
  int64_t HI = *in_h;
  int64_t WI = *in_w;
  int64_t CI = *in_ch;
  int64_t CO = *in_ch;
  int64_t NO = NI;
  int64_t HO = HI * 2.0;
  int64_t WO = WI * 2.0;
  *out_batch = NO;
  *out_h = HO;
  *out_w = WO;
  
  if (true) {
    
bilinear_upsampling_luncher(
    in_ptr,

    out_ptr,
    NI,
    HI,
    WI,
    CI,
    HO,
    WO,
    stream
);
return;
  }
  throw std::runtime_error(
      "Unsupported workload for this bilinear upsampling specialization."
  );
}