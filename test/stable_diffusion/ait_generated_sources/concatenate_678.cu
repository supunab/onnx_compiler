

#include <assert.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


#include <cuda_fp16.h>

        

#ifndef CHECK_ERROR_CAT
#define CHECK_ERROR_CAT(expr)                                \
  do {                                                       \
    cudaError_t status = (expr);                       \
    if (status != cudaSuccess) {                       \
      auto msg = std::string("Got error: ") +                \
        cudaGetErrorString(status) +                   \
        " at " + __FILE__ + ": " + std::to_string(__LINE__); \
      std::cerr << msg << std::endl;                         \
      throw std::runtime_error(msg);                         \
    }                                                        \
  } while (0)
#endif // CHECK_ERROR_CAT

#ifndef LAUNCH_CHECK_CAT
#define LAUNCH_CHECK_CAT() CHECK_ERROR_CAT(cudaGetLastError())
#endif // LAUNCH_CHECK_CAT



namespace {

//  Copyright (c) Meta Platforms, Inc. and affiliates.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
#ifndef AIT_TENSOR_ACCESSOR_CUH
#define AIT_TENSOR_ACCESSOR_CUH

// Returns a strided address based on a base pointer, an index and strided
// information.
// DATA_T: tensor data type.
// READ_T: actual data type used when reading data. e.g. for a "half"
// tensor, READ_T could be uint4 when all data is aligned.
// data: A base pointer in READ_T type.
// idx: read index in terms of READ_T.
// offset, original_total_elements_from_stride_dim and
// actual_total_elements_from_stride_dim are the corresponding data member
// values of TensorAccessor.
template <typename DATA_T, typename READ_T, bool is_contiguous>
__device__ __forceinline__ READ_T* get_strided_address(
    READ_T* data,
    int64_t idx,
    int64_t offset,
    int64_t original_total_elements_from_stride_dim,
    int64_t actual_total_elements_from_stride_dim) {
  (void)original_total_elements_from_stride_dim; // Suppress incorrect declared
                                                 // but never referenced warning
                                                 // from nvcc.
  (void)actual_total_elements_from_stride_dim; // Ditto.
  if constexpr (is_contiguous) {
    return reinterpret_cast<READ_T*>(reinterpret_cast<DATA_T*>(data) + offset) +
        idx;
  } else {
    constexpr int N_ELEMENTS_PER_READ = sizeof(READ_T) / sizeof(DATA_T);
    int64_t data_idx = idx * N_ELEMENTS_PER_READ;
    int64_t num_rows = data_idx / original_total_elements_from_stride_dim;
    int64_t row_offset = data_idx % original_total_elements_from_stride_dim;
    data_idx =
        num_rows * actual_total_elements_from_stride_dim + row_offset + offset;
    return reinterpret_cast<READ_T*>(
        reinterpret_cast<DATA_T*>(data) + data_idx);
  }
  return nullptr; // Suppress incorrect warning about missing return statement
                  // from nvcc.
}

static inline uint64_t max_power2_divisor(uint64_t n) {
  // max power of 2 which divides n
  return n & (~(n - 1));
}

// A TensorAccessor which handles strided tensor access underneath.
struct TensorAccessor {
  int64_t offset{0};
  bool is_contiguous{true};

  int stride_dim{-1};
  int64_t original_total_elements_from_stride_dim{-1};
  int64_t actual_total_elements_from_stride_dim{-1};

  // Returns an address based on a base pointer and an index.

  // DATA_T: tensor data type.
  // READ_T: actual data type used when reading data. e.g. for a "half"
  // tensor, READ_T could be uint4 when all data is aligned.
  // data: A base pointer in READ_T type.
  // idx: read index in terms of READ_T.
  template <typename DATA_T, typename READ_T>
  __device__ inline READ_T* get(READ_T* data, int64_t idx) const {
    return is_contiguous ? get_strided_address<DATA_T, READ_T, true>(
                               data,
                               idx,
                               offset,
                               original_total_elements_from_stride_dim,
                               actual_total_elements_from_stride_dim)
                         : get_strided_address<DATA_T, READ_T, false>(
                               data,
                               idx,
                               offset,
                               original_total_elements_from_stride_dim,
                               actual_total_elements_from_stride_dim);
  }

  uint64_t max_alignment() const {
    // gcd of max alignments
    auto alignment = max_power2_divisor(offset);
    if (!is_contiguous) {
      alignment |= max_power2_divisor(original_total_elements_from_stride_dim);
      alignment |= max_power2_divisor(actual_total_elements_from_stride_dim);
    }
    return max_power2_divisor(alignment);
  }

  bool is_valid_alignment(uint64_t n) const {
    // n is a power of 2; return whether tensor accessor alignment is divisible
    // by n.
    return !(max_alignment() & (n - 1));
  }
};

#endif


// TODO: support strided tensor with TensorAccessor
// For strided tensor, the index can be much larger than original if the stride is large
bool can_use_32bit_index_math(const int64_t elements, int64_t max_elem=std::numeric_limits<int32_t>::max()) {
  if (elements >= max_elem) {
    return false;
  }
  if (elements == 0) {
    return max_elem > 0;
  }

  return true;
}

template <typename T, int64_t NumInputs>
struct InputMetaData {
  const T *inputs[NumInputs]; /* pointer to each input */
  TensorAccessor input_accessors[NumInputs];
  int64_t concat_dim_offsets[NumInputs]; /* offset of each input along
                                            the concat dimension */
  int64_t concat_dim_values[NumInputs]; /* concat dimension value of
                                           each input */
  int64_t num_elems[NumInputs]; /* number of elements of each input */
};

template <int64_t Rank>
struct OutputMetaData {
  int64_t output_shape[Rank];
  int64_t output_strides[Rank];
};

__host__ __device__ __forceinline__
int64_t get_num_elems(const int64_t *shape, int64_t rank) {
  int64_t num = 1;
  for (int64_t i = 0; i < rank; i++) {
    num *= shape[i];
  }
  return num;
}

template <typename INDEX_T, int64_t Rank>
__host__ __device__ int64_t compute_output_elem_offset(
    const int64_t *output_shape,
    const int64_t *output_strides,
    const INDEX_T input_concat_dim_value,
    const INDEX_T concat_dim,
    INDEX_T linear_idx) {
  INDEX_T offset = 0;
  for (INDEX_T i = Rank - 1; i >= 1; --i) {
    INDEX_T cur_dim_size =
        i == concat_dim ? input_concat_dim_value : output_shape[i];
    INDEX_T next_dim_idx = linear_idx / cur_dim_size;
    INDEX_T cur_dim_idx = linear_idx - cur_dim_size * next_dim_idx;
    INDEX_T cur_dim_offset = cur_dim_idx * static_cast<INDEX_T>(output_strides[i]);
    offset += cur_dim_offset;
    linear_idx = next_dim_idx;
  }
  return offset + linear_idx * static_cast<INDEX_T>(output_strides[0]);
}
} // namespace

template <typename READ_T, typename ELEM_T, typename INDEX_T, int64_t Rank,
          int64_t NumInputs, int64_t ElemsPerThread>
__global__ void
concatenate_kernel(
    ELEM_T *orig_output,
    OutputMetaData<Rank> output_meta,
    InputMetaData<ELEM_T, NumInputs> input_meta,
    const INDEX_T concat_dim,
    const INDEX_T output_concat_dim_stride) {
  const INDEX_T tid = blockIdx.x * blockDim.x + threadIdx.x;
  const INDEX_T block_y = blockIdx.y % NumInputs;
  READ_T* output = reinterpret_cast<READ_T*>(orig_output);

  READ_T* input = const_cast<READ_T*>(
      reinterpret_cast<const READ_T*>(input_meta.inputs[block_y]));
  const TensorAccessor &input_accessor = input_meta.input_accessors[block_y];
  INDEX_T input_offset = input_meta.concat_dim_offsets[block_y];
  INDEX_T num_input_elems = input_meta.num_elems[block_y];
  INDEX_T input_concat_dim_value = input_meta.concat_dim_values[block_y];
  INDEX_T output_offset = input_offset * output_concat_dim_stride;

  constexpr unsigned read_t_sz = sizeof(READ_T);
  constexpr unsigned elem_t_sz = sizeof(ELEM_T);
  assert(read_t_sz >= elem_t_sz && (read_t_sz % elem_t_sz == 0));
  constexpr INDEX_T n_of_elem_t = read_t_sz / elem_t_sz;
  // number of READ_T elements per thread
  INDEX_T reads_per_thread_in_read_t = ElemsPerThread / n_of_elem_t;
  const INDEX_T num_elems_in_read_t = num_input_elems / n_of_elem_t;
  INDEX_T read_idx = tid;

#pragma unroll
  for (INDEX_T i = 0; i < reads_per_thread_in_read_t;
       i++, read_idx += blockDim.x * gridDim.x) {
    if (read_idx >= num_elems_in_read_t) {
      break;
    }
    READ_T tmp_v = *(input_accessor.get<ELEM_T, READ_T>(input, read_idx));
    /* make sure to adjust read_idx, which refers to location at
       (read_idx * n_of_elem_t) actually */

    INDEX_T output_elem_offset =
        compute_output_elem_offset<INDEX_T, Rank>(output_meta.output_shape,
                                                  output_meta.output_strides,
                                                  input_concat_dim_value,
                                                  concat_dim,
                                                  read_idx * n_of_elem_t);
    
    output[(output_offset + output_elem_offset) / n_of_elem_t] = tmp_v;
    
  }
}

enum class LoadVecType {
  VT_HALF = 0,
  VT_FLOAT,
  VT_FLOAT2,
  VT_FLOAT4
};

template <typename ELEM_T>
static inline LoadVecType get_vec_type(int64_t dim_size) {
  int64_t  size_elem_t = sizeof(ELEM_T);

#define HANDLE_ONE_VEC_TYPE(load_vec_type, vec_type)  \
  if (sizeof(vec_type) % size_elem_t == 0) {          \
    int64_t  n_of_elem_t = sizeof(vec_type) / size_elem_t; \
    if (dim_size % n_of_elem_t == 0) {                \
      return load_vec_type;                           \
    }                                                 \
  }

  HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT4, float4)
  HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT2, float2)
  HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT, float)
  HANDLE_ONE_VEC_TYPE(LoadVecType::VT_HALF, half)

#undef HANDLE_ONE_VEC_TYPE
  throw std::runtime_error(
      "Cannot resolve LoadVecType."
  );
}

template <typename ELEM_T, typename INDEX_T, int64_t Rank, int64_t NumInputs,
          int64_t ElemsPerThread, int64_t ThreadsPerBlock>
void concatenate_kernel_launcher(
    ELEM_T *output,
    const int64_t *output_shape,
    const ELEM_T *inputs[],
    const int64_t *real_input_shapes[],
    const TensorAccessor *input_accessors[],
    const int64_t concat_dim_offsets[],
    const int64_t concat_dim,
    LoadVecType min_vec_type,
    cudaStream_t stream) {

  OutputMetaData<Rank> output_meta;
  output_meta.output_strides[Rank - 1] = 1;
  output_meta.output_shape[Rank - 1] = output_shape[Rank - 1];
  for (INDEX_T i = Rank - 2; i >= 0; i--) {
    output_meta.output_strides[i] =
        output_meta.output_strides[i+1] * output_shape[i+1];
    output_meta.output_shape[i] = output_shape[i];
  }

  InputMetaData<ELEM_T, NumInputs> input_meta;
  INDEX_T max_num_input_elems = 0;
  for (INDEX_T i = 0; i < NumInputs; i++) {
    INDEX_T num_elems = get_num_elems(real_input_shapes[i], Rank);
    input_meta.inputs[i] = inputs[i];
    input_meta.input_accessors[i] = *(input_accessors[i]);
    input_meta.concat_dim_offsets[i] = concat_dim_offsets[i];
    input_meta.concat_dim_values[i] = real_input_shapes[i][concat_dim];
    input_meta.num_elems[i] = num_elems;

    max_num_input_elems = num_elems > max_num_input_elems ?
                          num_elems : max_num_input_elems;
  }

  constexpr INDEX_T elems_per_block = ThreadsPerBlock * ElemsPerThread;
  INDEX_T m = (max_num_input_elems % elems_per_block != 0);
  INDEX_T num_blocks_x =
      (max_num_input_elems / elems_per_block) + m;
  dim3 grid_config = dim3(static_cast<unsigned>(num_blocks_x), NumInputs);

#define HANDLE_ONE_VEC_TYPE(load_vec_type, vec_type)                        \
    case load_vec_type: {                                                   \
      if (ElemsPerThread * sizeof(ELEM_T) < sizeof(vec_type)) {             \
         throw std::runtime_error(                                          \
           std::string("No valid kernel available for ") + #vec_type);      \
      }                                                                     \
      concatenate_kernel<vec_type, ELEM_T, INDEX_T, Rank, NumInputs, ElemsPerThread> \
        <<<grid_config, ThreadsPerBlock, 0, stream>>>(                      \
            output,                                                         \
            output_meta,                                                    \
            input_meta,                                                     \
            concat_dim,                                                     \
            output_meta.output_strides[concat_dim]);                        \
      LAUNCH_CHECK_CAT();                                                   \
      break;                                                                \
    }

  switch (min_vec_type) {
    HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT4, float4)
    HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT2, float2)
    HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT, float)
    HANDLE_ONE_VEC_TYPE(LoadVecType::VT_HALF, half)
    default:
      throw std::runtime_error("Invalid LoadVecType\n");
  }

#undef HANDLE_ONE_VEC_TYPE
}

#undef CHECK_ERROR_CAT
#undef LAUNCH_CHECK_CAT

void concatenate_678(
    half *output,
    int64_t *output_shape[],
    const half *inputs[],
    const int64_t *real_input_shapes[], /* real_input_shapes, representing
                                 shapes of those inputs whose masks are False,
                                 i.e. inputs that will be copied to the output
                                 tensor by concat.*/
    const int64_t *all_input_shapes[], /* all_input_shapes include both
                                 kinds of inputs, i.e. no matter input_mask being
                                 True or False */
    const bool input_masks[],
    const int64_t concat_dim_sizes[],
    int64_t concat_dim,
    int64_t rank,
    int64_t num_real_inputs,
    int64_t num_all_inputs,
    cudaStream_t stream
    ) {

  if (rank <= 0) {
    throw std::runtime_error("rank must be larger than 0!");
  }
  if (concat_dim >= rank) {
    throw std::runtime_error("concat_dim must be smaller than rank!");
  }
  if (num_real_inputs < 1) {
    throw std::runtime_error("the number of inputs must >= 1!");
  }

  for (int64_t i = 0; i < rank; i++) {
    if (i == concat_dim) continue;
    int64_t dim = real_input_shapes[0][i];
    for (int64_t j = 1; j < num_real_inputs; j++) {
      if (real_input_shapes[j][i] != dim) {
        throw std::runtime_error(
          "invalid input shape, func_name: concatenate_678, dim: " +
          std::to_string(dim) + ", input_shape: " +
          std::to_string(real_input_shapes[j][i])
        );
      }
    }
  }

  int64_t output_concat_dim_value = 0;
  std::vector<int64_t> concat_dim_offsets;

  for (int64_t i = 0; i < num_all_inputs; i++) {
    if (input_masks[i]) {
      concat_dim_offsets.push_back(output_concat_dim_value);
    }
    output_concat_dim_value += concat_dim_sizes[i];
  }
  for (int64_t i = 0; i < rank; i++) {
    if (i == concat_dim) {
      *(output_shape[i]) = output_concat_dim_value;
    } else {
      *(output_shape[i]) = real_input_shapes[0][i];
    }
  }

  // If all input tensors are empty we are done
  bool empty = false;
  bool use_int32_index_math = true;
  for (int i = 0; i < num_real_inputs; i++) {
    int64_t num_elems = get_num_elems(real_input_shapes[i], rank);
    if (get_num_elems(real_input_shapes[i], rank) != 0) {
      empty = false;
      // make sure input is valid for each non-zero-size tensor
      if (!inputs[i]) {
        throw std::runtime_error("NULL input is found at: " + std::to_string(i));
      }
    }
    if (input_masks[i]) {
      use_int32_index_math &= can_use_32bit_index_math(num_elems);
    }
  }

  if (empty) {
    return;
  }

  // if the output has any zero dim size, we are done
  for (int i = 0; i < rank; i++) {
    if (*output_shape[i] == 0)
      return;
  }
  // make sure output is valid
  if (!output) {
    throw std::runtime_error("output is NULL!");
  }


  if (rank == 4 && num_real_inputs == 2) {



    TensorAccessor input_accessor0 = {
      0,
      
      true
      
      
    };
    TensorAccessor input_accessor1 = {
      0,
      
      true
      
      
    };

    const TensorAccessor *input_accessors[2] = {

      &input_accessor0, &input_accessor1

    };

    LoadVecType min_vec_type = LoadVecType::VT_FLOAT4;
    int64_t accessor_idx = 0;
    for (int64_t i = 0; i < num_all_inputs; i++) {
      int local_alignment;
      if (!input_masks[i] ||
          input_accessors[accessor_idx]->stride_dim == -1) {
        local_alignment = all_input_shapes[i][rank - 1];
        // int64_t is ok here because this happens on CPU
        for (int64_t j = rank - 2; j >= concat_dim; j--) {
          local_alignment *= all_input_shapes[i][j];
        }
      } else {
        local_alignment =
            input_accessors[accessor_idx]->max_alignment();
      }
      LoadVecType vec_type = get_vec_type<half>(local_alignment);
      min_vec_type = vec_type < min_vec_type ? vec_type : min_vec_type;
      if (input_masks[i]) {
        accessor_idx++;
      }
    }

    int64_t local_output_shape[] = {

      *(output_shape[0]),

      *(output_shape[1]),

      *(output_shape[2]),

      *(output_shape[3])
    };

  /* TODO: more profiling on ElemsPerThread and ThreadsPerBlock */
  if (use_int32_index_math) {
    concatenate_kernel_launcher<half,
                      int32_t,
                      4/*Rank*/,
                      2/*NumInputs*/,
                      16/*ElemsPerThread*/,
                      128/*THREADS_PER_BLOCK*/>(
      output, local_output_shape, inputs, real_input_shapes, input_accessors,
      concat_dim_offsets.data(), concat_dim, min_vec_type, stream);
  } else {
    concatenate_kernel_launcher<half,
                      int64_t,
                      4/*Rank*/,
                      2/*NumInputs*/,
                      16/*ElemsPerThread*/,
                      128/*THREADS_PER_BLOCK*/>(
      output, local_output_shape, inputs, real_input_shapes, input_accessors,
      concat_dim_offsets.data(), concat_dim, min_vec_type, stream);
  }
  return;
  }

  throw std::runtime_error(
      "Unsupported concat kernel specialization!"
  );
}