
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


#include "cutlass/layout/permute.h"

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



      
    using cutlass_tensorop_f16_s16816gemm_f16_128x64_32x6_tn_align8_base =
    cutlass::gemm::device::GemmUniversal<
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        cutlass::half_t, cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 64, 32>,
        cutlass::gemm::GemmShape<64, 32, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
          cutlass::half_t,
          8,
          float,
          float
        >,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
        6,
        8,
        8,
        cutlass::arch::OpMultiplyAdd,
        cutlass::ComplexTransform::kNone,
        cutlass::ComplexTransform::kNone,
        false,  /*GatherA*/
        false,  /*GatherB*/
        false,  /*ScatterD*/
        cutlass::layout::Tensor5DPermute20314<64, 1, 8> /*PermuteDLayout*/
    >;
      
using fd9106d7291c48fc10faca140108b9deb185eed00 = cutlass_tensorop_f16_s16816gemm_f16_128x64_32x6_tn_align8_base;

void gemm_rcr_permute_446 (
    cutlass::half_t* a_ptr,
    cutlass::half_t* b_ptr,
    cutlass::half_t* c_ptr,
    uint8_t* workspace,
    int split_k,
    int64_t* a_dim0,
    int64_t* a_dim1,
    int64_t* b_dim0,
    int64_t* b_dim1,
    int64_t* c_dim0,
    int64_t* c_dim1,
    cudaStream_t stream
  ) {
  
 int64_t M = (*a_dim0);

 int64_t N = (*b_dim0);

 int64_t K = (*a_dim1);
  
  
  
  int64_t output_stride = N;
  int64_t output_offset = 0;
  
    
  
  
  int64_t a_size = 1;

    a_size *= *a_dim0;

    a_size *= *a_dim1;

  if (a_size != 0 && !a_ptr) {
    throw std::runtime_error("input a is null!");
  }

  int64_t b_size = 1;

    b_size *= *b_dim0;

    b_size *= *b_dim1;

  if (b_size != 0 && !b_ptr) {
    throw std::runtime_error("input b is null!");
  }

  int64_t c_size = 1;

    c_size *= *c_dim0;

    c_size *= *c_dim1;

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

  
  if (M == 128 && N == 1280 && K == 1280) {
    
//  TODO: cast to right dtype
    using ElementComputeEpilogue = typename fd9106d7291c48fc10faca140108b9deb185eed00::ElementAccumulator;

    typename fd9106d7291c48fc10faca140108b9deb185eed00::Arguments arguments{


    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    split_k,
    {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},
    (void*) a_ptr,
    (void*) b_ptr,
    (void*) c_ptr,
    (void*) (c_ptr + output_offset),
    M * K,
    N * K,
    M * N,
    M * N,
    K,
    K,
    N,
    output_stride

    };
    fd9106d7291c48fc10faca140108b9deb185eed00 gemm_op;

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
      std::cout << "weight_ndims0: " << *b_dim0 << std::endl;
      std::cout << "weight_ndims1: " << *b_dim1 << std::endl;
      std::cout << "output_ndims0: " << *c_dim0 << std::endl;
      std::cout << "output_ndims1: " << *c_dim1 << std::endl;
  throw std::runtime_error(
      "Unsupported workload for this gemm_rcr_permute_446 specialization."
  );
}