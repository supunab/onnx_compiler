
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/device_memory.h"



#define CUTLASS_CHECK(status)                                                         \
  {                                                                                   \
    cutlass::Status error = status;                                                   \
    if (error != cutlass::Status::kSuccess) {                                         \
      auto msg = std::string("[") + __FILE__ + "] Got cutlass error: " +              \
          cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);         \
      std::cerr << msg << std::endl;                                                  \
      throw std::runtime_error(msg);                                                  \
    }                                                                                 \
  }



  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using f1e187c1d7dd787f03c34a14ef4a2c185e56772ea = Operation_cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tn_align8;

void bmm_rcr_303 (
    cutlass::half_t* a_ptr,
    cutlass::half_t* b_ptr,
    cutlass::half_t* c_ptr,
    uint8_t* workspace,
    int64_t* a_dim0,
    int64_t* a_dim1,
    int64_t* a_dim2,
    int64_t* b_dim0,
    int64_t* b_dim1,
    int64_t* b_dim2,
    int64_t* c_dim0,
    int64_t* c_dim1,
    int64_t* c_dim2,
    cudaStream_t stream
  ) {
  
 int64_t B = (*a_dim0);

 int64_t M = (*a_dim1);

 int64_t N = (*b_dim1);

 int64_t K = (*a_dim2);
  
  int64_t input_a_batch_stride = M * K;
  int64_t input_a_stride = K;
  int64_t input_a_offset = 0; // default to 0
  int64_t input_b_batch_stride = N * K;
  int64_t input_b_stride = K;
  int64_t input_b_offset = 0; // default to 0
    
  
  int64_t output_batch_stride = M * N;
  int64_t output_stride = N;
  int64_t output_offset = 0; // default to 0
    
  
  
  int64_t a_size = 1;

    a_size *= *a_dim0;

    a_size *= *a_dim1;

    a_size *= *a_dim2;

  if (a_size != 0 && !a_ptr) {
    throw std::runtime_error("input a is null!");
  }

  int64_t b_size = 1;

    b_size *= *b_dim0;

    b_size *= *b_dim1;

    b_size *= *b_dim2;

  if (b_size != 0 && !b_ptr) {
    throw std::runtime_error("input b is null!");
  }

  int64_t c_size = 1;

    c_size *= *c_dim0;

    c_size *= *c_dim1;

    c_size *= *c_dim2;

  if (c_size != 0) {
    if (!c_ptr) {
      throw std::runtime_error("input c is null!");
    }
  } else {
    // output is empty and safe to return
    return;
  }

  // One of the input tensor are empty
  if (a_size == 0 || b_size == 0) {
    return;
  }

  
  if (B == 16 && M == 256 && N == 256 && K == 160) {
    
//  TODO: cast to right dtype
    using ElementComputeEpilogue = typename f1e187c1d7dd787f03c34a14ef4a2c185e56772ea::ElementAccumulator;

    typename f1e187c1d7dd787f03c34a14ef4a2c185e56772ea::Arguments arguments{


    cutlass::gemm::GemmUniversalMode::kBatched,
    {M, N, K},
    B,
    {ElementComputeEpilogue(1.0), ElementComputeEpilogue(0)},
    (void*) (a_ptr + input_a_offset),
    (void*) (b_ptr + input_b_offset),
    (void*) (c_ptr + output_offset),
    (void*) (c_ptr + output_offset),
    input_a_batch_stride,
    input_b_batch_stride,
    output_batch_stride,
    output_batch_stride,
    input_a_stride,
    input_b_stride,
    output_stride,
    output_stride

    };
    f1e187c1d7dd787f03c34a14ef4a2c185e56772ea gemm_op;

    auto status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);
    status = gemm_op.initialize(arguments, workspace, stream);
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
    return;
  }
      std::cout << "input_ndims0: " << *a_dim0 << std::endl;
      std::cout << "input_ndims1: " << *a_dim1 << std::endl;
      std::cout << "input_ndims2: " << *a_dim2 << std::endl;
      std::cout << "weight_ndims0: " << *b_dim0 << std::endl;
      std::cout << "weight_ndims1: " << *b_dim1 << std::endl;
      std::cout << "weight_ndims2: " << *b_dim2 << std::endl;
      std::cout << "output_ndims0: " << *c_dim0 << std::endl;
      std::cout << "output_ndims1: " << *c_dim1 << std::endl;
      std::cout << "output_ndims2: " << *c_dim2 << std::endl;
  throw std::runtime_error(
      "Unsupported workload for this bmm_rcr_303 specialization."
  );
}