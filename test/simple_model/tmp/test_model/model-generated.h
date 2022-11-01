
#pragma once
#include "logging.h"
#include "device_functions-generated.h"
#include "model_interface.h"
#include "raii_wrapper.h"
#include "macros.h"
#include <algorithm>
#include <deque>
#include <string>
#include <unordered_map>
#include <math.h>


void gemm_rcr_bias_relu_8(
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
  uint8_t*,

    int,


  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,

  cudaStream_t
);

void gemm_rcr_bias_relu_9(
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
  uint8_t*,

    int,


  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,

  cudaStream_t
);

void gemm_rcr_bias_relu_10(
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
  uint8_t*,

    int,


  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,

  cudaStream_t
);

void gemm_rcr_bias_6(
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
  uint8_t*,

    int,


  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,

  cudaStream_t
);

    void softmax_7(cutlass::half_t* input,
                   cutlass::half_t* output,

                   int64_t* in_0,

                   cudaStream_t stream);
    

#define CHECK_VECTOR_ACCESS(vector, idx)                                  \
  if (idx >= vector.size()) {                                             \
    throw std::out_of_range(                                              \
        "[__func__]: index out of range, " #vector ".size()=" +           \
        std::to_string(vector.size()) + ", got " + std::to_string(idx));  \
  }

namespace ait {
namespace {
void DeviceCheckLastError(const char* file, int line) {
  auto device_error = GetLastError();
  if (device_error != GetDeviceSuccess()) {
    std::string msg = std::string("Got error: ") + GetLastErrorString() +
                      " enum: " + std::to_string(device_error) +
                      " at " + file + ": " + std::to_string(line);
    LOG(ERROR) << msg;
    throw std::runtime_error(msg);
  }
}
}

// Model is the class that actually performs inference. It owns memory for
// intermediate tensors and dynamic dimensions. Constants are owned by
// the model's owning container object, and input/output memory is owned
// by the user.
// Once an inference run has started, it is not safe to re-use the Model
// until the run has finished!
class Model {
  public:
  Model(
      size_t blob_size,
      size_t workspace_size,
      size_t num_inputs,
      size_t num_outputs,
      size_t num_unbound_constants,
      uint8_t* constants)
      : blob(RAII_DeviceMalloc(blob_size)),
        workspace(RAII_DeviceMalloc(workspace_size)),
        params(num_inputs + num_outputs + num_unbound_constants),
        num_inputs(num_inputs),
        constants(constants) {
      dmlc::InitLogging("aitemplate"); // TODO(xxx): render network name
      // LOG(INFO) << "Init AITemplate Runtime."; // TODO(supuna): removed logging init message
      global_workspace = static_cast<uint8_t*>(workspace.get()) + 0;
      unique_workspace = static_cast<uint8_t*>(workspace.get());
      DEVICE_CHECK(GetDevice(&device_idx))
      DEVICE_CHECK(CreateEvent(&run_finished));
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
      DEVICE_CHECK(cudaDeviceGetAttribute(
        &max_smem_size, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_idx));
#endif
      DEVICE_CHECK(GetDeviceProperties(&device_properties, device_idx));
      DEVICE_CHECK(StreamCreate(&graph_capture_stream, /*non_blocking=*/true));

       constant_name_to_ptr_["layers_0_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layers_0_weight));
     constant_name_to_ptr_["layers_0_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layers_0_bias));
     constant_name_to_ptr_["layers_1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layers_1_weight));
     constant_name_to_ptr_["layers_1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layers_1_bias));
     constant_name_to_ptr_["layers_2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layers_2_weight));
     constant_name_to_ptr_["layers_2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layers_2_bias));
     constant_name_to_ptr_["layers_3_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layers_3_weight));
     constant_name_to_ptr_["layers_3_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layers_3_bias));
      auto* blob_ptr = static_cast<uint8_t*>(blob.get());
      onnx_Gemm_10 = reinterpret_cast<decltype(onnx_Gemm_10)>(blob_ptr + 4194304);
    onnx_Gemm_12 = reinterpret_cast<decltype(onnx_Gemm_12)>(blob_ptr + 0);
    onnx_Gemm_14 = reinterpret_cast<decltype(onnx_Gemm_14)>(blob_ptr + 4194304);
    hidden_states_11 = reinterpret_cast<decltype(hidden_states_11)>(blob_ptr + 0);
  
       params[0].shape_ptrs = {ParamDim(1024, 1024, &onnx_Gemm_0_dim_0), ParamDim(784, 784, &onnx_Gemm_0_dim_1)};
     params[2].shape_ptrs = {ParamDim(1024, 1024, &layers_0_weight_dim_0), ParamDim(784, 784, &layers_0_weight_dim_1)};
     params[3].shape_ptrs = {ParamDim(1024, 1024, &layers_0_bias_dim_0)};
     params[4].shape_ptrs = {ParamDim(2048, 2048, &layers_1_weight_dim_0), ParamDim(1024, 1024, &layers_1_weight_dim_1)};
     params[5].shape_ptrs = {ParamDim(2048, 2048, &layers_1_bias_dim_0)};
     params[6].shape_ptrs = {ParamDim(512, 512, &layers_2_weight_dim_0), ParamDim(2048, 2048, &layers_2_weight_dim_1)};
     params[7].shape_ptrs = {ParamDim(512, 512, &layers_2_bias_dim_0)};
     params[8].shape_ptrs = {ParamDim(10, 10, &layers_3_weight_dim_0), ParamDim(512, 512, &layers_3_weight_dim_1)};
     params[9].shape_ptrs = {ParamDim(10, 10, &layers_3_bias_dim_0)};
     params[1].shape_ptrs = {ParamDim(1024, 1024, &onnx_Gemm_0_dim_0), ParamDim(10, 10, &layers_3_weight_dim_0)};
    }

    ~Model() {
      DestroyEvent(run_finished);
      StreamDestroy(graph_capture_stream);
      if (graph_exec != nullptr) {
        GraphExecDestroy(graph_exec);
      }
      if (graph != nullptr) {
        GraphDestroy(graph);
      }
    }

    Model(Model&&) = default;
    Model& operator=(Model&&) = default;

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    void SetUpInputsOutputs() {
             onnx_Gemm_0 = static_cast<decltype(onnx_Gemm_0)>(params[0].ptr);

if (onnx_Gemm_0 == nullptr) {
    throw std::runtime_error("Constant onnx_Gemm_0 was not set! Set the value with set_constant.");
}
    

if (layers_0_weight == nullptr) {
    throw std::runtime_error("Constant layers_0_weight was not set! Set the value with set_constant.");
}
    

if (layers_0_bias == nullptr) {
    throw std::runtime_error("Constant layers_0_bias was not set! Set the value with set_constant.");
}
    

if (layers_1_weight == nullptr) {
    throw std::runtime_error("Constant layers_1_weight was not set! Set the value with set_constant.");
}
    

if (layers_1_bias == nullptr) {
    throw std::runtime_error("Constant layers_1_bias was not set! Set the value with set_constant.");
}
    

if (layers_2_weight == nullptr) {
    throw std::runtime_error("Constant layers_2_weight was not set! Set the value with set_constant.");
}
    

if (layers_2_bias == nullptr) {
    throw std::runtime_error("Constant layers_2_bias was not set! Set the value with set_constant.");
}
    

if (layers_3_weight == nullptr) {
    throw std::runtime_error("Constant layers_3_weight was not set! Set the value with set_constant.");
}
    

if (layers_3_bias == nullptr) {
    throw std::runtime_error("Constant layers_3_bias was not set! Set the value with set_constant.");
}
    
     _16 = static_cast<decltype(_16)>(params[1].ptr);

if (_16 == nullptr) {
    throw std::runtime_error("Constant _16 was not set! Set the value with set_constant.");
}
    
    }

    void DeviceToDeviceCopies(StreamType stream) {
  
    }

    void Run(StreamType stream, bool graph_mode) {
      SetUpInputsOutputs();
      if (target_has_graph_mode && graph_mode) {
        RunAsGraph(stream);
      } else {
        RunImpl(stream);
      }
      DEVICE_CHECK(EventRecord(run_finished, stream));
    }

    void RunImpl(StreamType stream) {
  
  
    {
    

    gemm_rcr_bias_relu_8(
        onnx_Gemm_0,
        layers_0_weight,

        layers_0_bias,

        onnx_Gemm_10,
        global_workspace,
        1,

        &onnx_Gemm_0_dim_0,

        &onnx_Gemm_0_dim_1,


        &layers_0_weight_dim_0,

        &layers_0_weight_dim_1,


        &onnx_Gemm_0_dim_0,

        &layers_0_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_relu_9(
        onnx_Gemm_10,
        layers_1_weight,

        layers_1_bias,

        onnx_Gemm_12,
        global_workspace,
        1,

        &onnx_Gemm_0_dim_0,

        &layers_0_weight_dim_0,


        &layers_1_weight_dim_0,

        &layers_1_weight_dim_1,


        &onnx_Gemm_0_dim_0,

        &layers_1_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_relu_10(
        onnx_Gemm_12,
        layers_2_weight,

        layers_2_bias,

        onnx_Gemm_14,
        global_workspace,
        1,

        &onnx_Gemm_0_dim_0,

        &layers_1_weight_dim_0,


        &layers_2_weight_dim_0,

        &layers_2_weight_dim_1,


        &onnx_Gemm_0_dim_0,

        &layers_2_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_6(
        onnx_Gemm_14,
        layers_3_weight,

        layers_3_bias,

        hidden_states_11,
        global_workspace,
        1,

        &onnx_Gemm_0_dim_0,

        &layers_2_weight_dim_0,


        &layers_3_weight_dim_0,

        &layers_3_weight_dim_1,


        &onnx_Gemm_0_dim_0,

        &layers_3_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    softmax_7(
       reinterpret_cast<cutlass::half_t*>(&(hidden_states_11->raw())),
       reinterpret_cast<cutlass::half_t*>(&(_16->raw())),

        &onnx_Gemm_0_dim_0,

       stream
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
      DeviceToDeviceCopies(stream);
    }

    bool IsPending() {
      auto query = QueryEvent(run_finished);
      if (query == GetDeviceNotReady()) {
        return true;
      }
      if (query != GetDeviceSuccess()) {
        LOG(WARNING) << "Pending model run did not finish successfully. Error: "
                    << GetErrorString(query);
      }
      return false;
    }

    void WaitForCompletion() {
      DEVICE_CHECK(EventSynchronize(run_finished));
    }

    size_t NumInputs() const {
      return num_inputs;
    }

    size_t NumOutputs() const {
      return params.size() - num_inputs;
    }

    void SetParam(const void* src, size_t param_idx) {
      CHECK_VECTOR_ACCESS(params, param_idx)
      // const_cast is not ideal here, but it is unfortunately
      // necessary:
      // 1) We store outputs and inputs in the same vector,
      //    and outputs cannot be const.
      // 2) Most of the codegen is not const-correct (most ops
      //    require non-const pointers). So even if we put const
      //    pointers into params, a const_cast would be required
      //    somewhere else.
      params[param_idx].ptr = const_cast<void*>(src);
    }

    void SetInput(const void* src, const AITemplateParamShape& shape, size_t idx) {
      SetInputShape(shape, idx);
      SetParam(src, idx);
    }

    void SetOutput(void* src, size_t idx) {
      SetParam(src, idx + num_inputs);
    }

    // Write the (possibly dynamic) output shape to the given pointer.
    // Note that this should be called _after_ the shape inference in
    // Run() is finished. output_shape_out should be able to store
    // at least GetOutputMaximumShape(idx).size values.
    void GetOutputShape(size_t idx, int64_t* output_shape_out) {
      const auto param_idx = idx + num_inputs;
      CHECK_VECTOR_ACCESS(params, param_idx);
      const auto& shape_ptrs = params[param_idx].shape_ptrs;
      for (size_t i = 0; i < shape_ptrs.size(); ++i) {
        output_shape_out[i] = shape_ptrs[i].GetValue();
      }
    }

    void SetConstant(const char* name, const void* src) {
      auto it = constant_name_to_ptr_.find(name);
      if (it == constant_name_to_ptr_.end()) {
        throw std::out_of_range(std::string("Could not find constant ") + name);
      }
      const void** ptr = it->second;
      *ptr = src;
    }

  private:
    void SetInputShape(const AITemplateParamShape& shape, size_t idx) {
      auto& param = params[idx];
      if (shape.size != param.shape_ptrs.size()) {
        throw std::runtime_error(
          "[SetInputShape] Got wrong param shape for input " + std::to_string(idx) +
          "; expected " + std::to_string(param.shape_ptrs.size()) + ", got " +
          std::to_string(shape.size));
      }
      for (size_t i = 0; i < param.shape_ptrs.size(); ++i) {
        param.shape_ptrs[i].SetValue(shape.shape_data[i]);
      }
    }

    void RunAsGraph(StreamType stream) {
      DEVICE_CHECK(StreamBeginCapture(graph_capture_stream));
      try {
        RunImpl(graph_capture_stream);
      } catch (...) {
        DEVICE_CHECK(StreamEndCapture(graph_capture_stream, &graph));
        throw;
      }
      DEVICE_CHECK(StreamEndCapture(graph_capture_stream, &graph));

      if (graph_exec == nullptr) {
        DEVICE_CHECK(GraphInstantiate(&graph_exec, graph));
      } else if (GraphExecUpdate(graph_exec, graph) != GetDeviceSuccess()) {
        DEVICE_CHECK(GraphExecDestroy(graph_exec));
        DEVICE_CHECK(GraphInstantiate(&graph_exec, graph));
      }

      DEVICE_CHECK(GraphExecLaunch(graph_exec, stream));
    }

    int device_idx;
    int max_smem_size{0};
    DevicePropertyType device_properties;
    // This event tracks when the inference is finished
    // so that this Model may be reclaimed by its owning
    // ModelContainer.
    EventType run_finished;
    // A blob of memory used for storing intermediate tensors.
    GPUPtr blob;
    // Memory for constants that were folded into the *.so. Unowned by Model,
    // owned by ModelContainer.
    // TODO: make this const. It can't be const right now because we derive
    // tensor pointers from it, and no tensor pointers are const.
    uint8_t* constants;
    size_t num_inputs;

    // The workspace blob is used as scratch memory. See
    // _generate_workspace in memory planning for more information.
    GPUPtr workspace;
    uint8_t* global_workspace{nullptr};
    uint8_t* unique_workspace{nullptr};

    class ParamDim {
      public:
        ParamDim(int64_t lower_bound, int64_t upper_bound, int64_t* value) :
          lower_bound_(lower_bound),
          upper_bound_(upper_bound),
          value_(value) {}

        void SetValue(int64_t new_value) {
          if (new_value < lower_bound_ || new_value > upper_bound_) {
            throw std::out_of_range(
              "[SetValue] Dimension got value out of bounds; expected value to be in [" +
              std::to_string(lower_bound_) + ", " + std::to_string(upper_bound_) + "], but got " +
              std::to_string(new_value)
            );
          }
          *value_ = new_value;
        }

        int64_t GetValue() const {
          return *value_;
        }

      private:
        int64_t lower_bound_;
        int64_t upper_bound_;
        int64_t* value_;
    };

    struct ParamInfo {
      void* ptr = nullptr;
      // TODO add offset
      const char* name;
      std::vector<ParamDim> shape_ptrs;
    };

    // Contains info for all tensors marked as inputs
    // or outputs. The first num_inputs elements are the inputs.
    // Constants are not included.
    std::vector<ParamInfo> params;

    GraphExecType graph_exec = nullptr;
    GraphType graph = nullptr;
    StreamType graph_capture_stream;

    std::unordered_map<std::string, const void**> constant_name_to_ptr_;

    constexpr static bool target_has_graph_mode = true;

   cutlass::half_t* onnx_Gemm_0 {nullptr};
   cutlass::half_t* layers_0_weight {nullptr};
   cutlass::half_t* layers_0_bias {nullptr};
   cutlass::half_t* onnx_Gemm_10 {nullptr};
   cutlass::half_t* layers_1_weight {nullptr};
   cutlass::half_t* layers_1_bias {nullptr};
   cutlass::half_t* onnx_Gemm_12 {nullptr};
   cutlass::half_t* layers_2_weight {nullptr};
   cutlass::half_t* layers_2_bias {nullptr};
   cutlass::half_t* onnx_Gemm_14 {nullptr};
   cutlass::half_t* layers_3_weight {nullptr};
   cutlass::half_t* layers_3_bias {nullptr};
   cutlass::half_t* hidden_states_11 {nullptr};
   cutlass::half_t* _16 {nullptr};
   int64_t onnx_Gemm_0_dim_0 { 1024 };
   int64_t onnx_Gemm_0_dim_1 { 784 };
   int64_t layers_0_weight_dim_0 { 1024 };
   int64_t layers_0_weight_dim_1 { 784 };
   int64_t layers_0_bias_dim_0 { 1024 };
   int64_t layers_1_weight_dim_0 { 2048 };
   int64_t layers_1_weight_dim_1 { 1024 };
   int64_t layers_1_bias_dim_0 { 2048 };
   int64_t layers_2_weight_dim_0 { 512 };
   int64_t layers_2_weight_dim_1 { 2048 };
   int64_t layers_2_bias_dim_0 { 512 };
   int64_t layers_3_weight_dim_0 { 10 };
   int64_t layers_3_weight_dim_1 { 512 };
   int64_t layers_3_bias_dim_0 { 10 };

};
} // namespace ait