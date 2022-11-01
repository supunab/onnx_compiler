
#include "model_container.h"
#include "owned_constants.h"

namespace ait {
namespace {
// Contains the metadata for each constant.
constexpr std::array<ConstantInfo, 0> owned_constants = {
  
};
} // namespace

ModelContainerBase::ModelContainerBase(
    size_t num_inputs,
    size_t num_outputs,
    size_t num_unbound_constants,
    size_t params_size)
    : constants_(RAII_DeviceMalloc(params_size)),
      num_params_(num_inputs + num_outputs + num_unbound_constants),
      param_names_(num_params_),
      param_dtypes_(num_params_),
      max_param_shapes_(num_params_),
      max_param_numel_(num_params_),
      max_param_storage_bytes_(num_params_) {
     unbound_constant_name_to_idx_["layers_0_weight"] = 0;
     unbound_constant_name_to_idx_["layers_0_bias"] = 1;
     unbound_constant_name_to_idx_["layers_1_weight"] = 2;
     unbound_constant_name_to_idx_["layers_1_bias"] = 3;
     unbound_constant_name_to_idx_["layers_2_weight"] = 4;
     unbound_constant_name_to_idx_["layers_2_bias"] = 5;
     unbound_constant_name_to_idx_["layers_3_weight"] = 6;
     unbound_constant_name_to_idx_["layers_3_bias"] = 7;
     param_names_[0] = "onnx_Gemm_0";
     param_names_[2] = "layers_0_weight";
     param_names_[3] = "layers_0_bias";
     param_names_[4] = "layers_1_weight";
     param_names_[5] = "layers_1_bias";
     param_names_[6] = "layers_2_weight";
     param_names_[7] = "layers_2_bias";
     param_names_[8] = "layers_3_weight";
     param_names_[9] = "layers_3_bias";
     param_names_[1] = "_16";
     param_dtypes_[0] = AITemplateDtype::kHalf;
     param_dtypes_[2] = AITemplateDtype::kHalf;
     param_dtypes_[3] = AITemplateDtype::kHalf;
     param_dtypes_[4] = AITemplateDtype::kHalf;
     param_dtypes_[5] = AITemplateDtype::kHalf;
     param_dtypes_[6] = AITemplateDtype::kHalf;
     param_dtypes_[7] = AITemplateDtype::kHalf;
     param_dtypes_[8] = AITemplateDtype::kHalf;
     param_dtypes_[9] = AITemplateDtype::kHalf;
     param_dtypes_[1] = AITemplateDtype::kHalf;
     max_param_shapes_[0] = {1024, 784};
     max_param_shapes_[2] = {1024, 784};
     max_param_shapes_[3] = {1024};
     max_param_shapes_[4] = {2048, 1024};
     max_param_shapes_[5] = {2048};
     max_param_shapes_[6] = {512, 2048};
     max_param_shapes_[7] = {512};
     max_param_shapes_[8] = {10, 512};
     max_param_shapes_[9] = {10};
     max_param_shapes_[1] = {1024, 10};
  for (size_t i = 0; i < num_params_; ++i) {
    max_param_numel_[i] = std::accumulate(
      max_param_shapes_[i].begin(),
      max_param_shapes_[i].end(),
      1,
      std::multiplies<int64_t>()
    );
    max_param_storage_bytes_[i] = max_param_numel_[i] * AITemplateDtypeSizeBytes(param_dtypes_[i]);
  }

  auto* constants_ptr = static_cast<uint8_t*>(constants_.get());
  DEVICE_CHECK(DeviceMemset(constants_ptr, 0, params_size));
  const auto binary_constants_bin_size = static_cast<size_t>(_binary_constants_bin_end - _binary_constants_bin_start);
  for (auto& constant_info : owned_constants) {
    auto* dst = constants_ptr + constant_info.internal_offset;
    if (constant_info.data_offset + constant_info.num_bytes > binary_constants_bin_size) {
      throw std::runtime_error(std::string("Copying constant ") + constant_info.name + " would overflow constant buffer");
    }
    DEVICE_CHECK(CopyToDevice(dst, _binary_constants_bin_start + constant_info.data_offset, constant_info.num_bytes));
  }
}

ModelContainer* CreateModelContainer(size_t num_runtimes) {
  // num_runtimes, blob_size, workspace_size, num_inputs, num_outputs, num_unbound_constants, param_size
  return new ModelContainer(num_runtimes, 6291456, 0, 1, 1, 8, 0);
}
} // namespace ait