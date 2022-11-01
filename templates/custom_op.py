import jinja2

CUSTOM_OP_HEADER = jinja2.Template(
    """
#pragma once
#include "onnxruntime_c_api.h"

#ifdef __GNUC__ // Applies to any compiler with GNU extensions (clang and g++)
#define AIT_EXPORT __attribute__((__visibility__("default")))
#else
#ifdef _WIN32
#define AIT_EXPORT __declspec(dllexport)
#else
#define AIT_EXPORT
#endif
#endif

#ifdef __cplusplus
extern "C"
{
#endif
    // TODO(supuna): why ORT_EXPORT didn't work!!!
    AIT_EXPORT OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api);

#ifdef __cplusplus
}
#endif
    """
)

CUSTOM_OP_SOURCE = jinja2.Template(
    """
#include "ort_ait_custom_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <vector>
#include <cmath>
#include <mutex>
#include <unistd.h>


#include "model_interface.h"

static const char *c_OpDomain = "ait.customop";

// below are basically 
struct OrtCustomOpDomainDeleter {
  explicit OrtCustomOpDomainDeleter(const OrtApi *ort_api) {
    ort_api_ = ort_api;
  }
  void operator()(OrtCustomOpDomain *domain) const {
    ort_api_->ReleaseCustomOpDomain(domain);
  }

  const OrtApi *ort_api_;
};

using OrtCustomOpDomainUniquePtr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>;
static std::vector<OrtCustomOpDomainUniquePtr> ort_custom_op_domain_container;
static std::mutex ort_custom_op_domain_mutex;

// TODO: what is this? why this is needed?
static void AddOrtCustomOpDomainToContainer(OrtCustomOpDomain *domain, const OrtApi *ort_api)
{
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  auto ptr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>(domain, OrtCustomOpDomainDeleter(ort_api));
  ort_custom_op_domain_container.push_back(std::move(ptr));
}

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

struct AITKernel {
  AITKernel(const OrtApi& api)
      : ort_(api) {
  }

  void Compute(OrtKernelContext* context) {
    {{get_input_ort_values_body}}
    {{get_ort_value_info_body}}
    {{get_input_shapes_body}}
    // unsafe const --> non-const cast to satisfy AITData requiring a non-const
    {{get_input_data_ptr_body}}

    // Setup output
    {{output_shapes_body}}
    {{get_output_ort_values_body}}
    {{get_output_data_ptr_body}}

    /* Calling AITemplate generated code */
    // init AITData objects from data
    {{init_ait_data_body}}

    AITemplateModelHandle handle;
    AITemplateModelContainerCreate(&handle, 1);

    {{set_ait_constants_body}}
    {{set_ait_inputs_body}}
    {{set_ait_outputs_body}}
    {{set_ait_output_shapes_body}}

    // obtain and assign the cuda stream
    auto stream_handle = reinterpret_cast<AITemplateStreamHandle>(ort_.KernelContext_GetGPUComputeStream(context));
    
    // TODO(supuna): do we need sync = true? does ort assume that the result is computed when it's returned from a kernel?
    AITemplateModelContainerRun(handle, ait_inputs, {{num_inputs}}, ait_outputs, {{num_outputs}}, stream_handle, /* sync */ true, true, ait_output_shapes);
  }

 private:
  Ort::CustomOpApi ort_;
};


// define the CustomOp 
struct AITModelOp : Ort::CustomOpBase<AITModelOp, AITKernel> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* /* info */) const {
    return new AITKernel(api);
  }; 

  const char* GetName() const { return "AITModelOp"; };

  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; };

  size_t GetInputTypeCount() const { return {{custom_op_input_count}}; }; 
  ONNXTensorElementDataType GetInputType(size_t i /*index*/) const { 
    {{get_input_type_body}}
  };

  size_t GetOutputTypeCount() const { return {{custom_op_output_count}}; };
  ONNXTensorElementDataType GetOutputType(size_t i /*index*/) const {
    {{get_output_type_body}}
  };

} c_AITModelOp;

// this is the entry point that gets called when this is loaded as a shared library
OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  OrtCustomOpDomain* domain = nullptr;
  const OrtApi* ortApi = api->GetApi(ORT_API_VERSION); 

  if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) {
    return status;
  }

  AddOrtCustomOpDomainToContainer(domain, ortApi);

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_AITModelOp)) {
    return status;
  }

  return ortApi->AddCustomOpDomain(options, domain);
}

    """
)

GET_INPUT_ORT_VALUES_BODY_LINE = jinja2.Template(
    """const OrtValue* {{name}}_ort = ort_.KernelContext_GetInput(context, {{input_id}});"""
)

GET_ORT_VALUE_INFO_BODY_LINE = jinja2.Template(
    """auto {{name}}_ort_info = ort_.GetTensorTypeAndShape({{name}}_ort);"""
)

GET_INPUT_SHAPES_BODY_LINE = jinja2.Template(
    """auto {{name}}_shape = ort_.GetTensorShape({{name}}_ort_info);"""
)

GET_INPUT_DATA_PTR_BODY_LINE = jinja2.Template(
    """auto {{name}}_data = const_cast<void*>(ort_.GetTensorData<void>({{name}}_ort));"""
)

#TODO: use proper jinja2 to replace {{shape}} with a list of args
OUTPUT_SHAPES_BODY_LINE = jinja2.Template(
    """
const int {{name}}_rank = {{rank}};
int64_t {{name}}_shape_data[{{rank}}] = { {{shape}} };
    """
)

GET_OUTPUT_ORT_VALUES_BODY_LINE = jinja2.Template(
    """OrtValue* {{name}}_ort = ort_.KernelContext_GetOutput(context, {{output_id}}, {{name}}_shape_data, {{name}}_rank);"""
)

GET_OUTPUT_DATA_PTR_BODY_LINE = jinja2.Template(
    """void* {{name}}_data = ort_.GetTensorMutableData<void>({{name}}_ort);"""
)

# TODO(supuna): dtype hardcoded in AIT
INIT_AIT_DATA_INPUT_BODY_LINE = jinja2.Template(
    """
auto {{name}}_shape_ait = AITemplateParamShape({{name}}_shape.data(), {{name}}_shape.size());
auto {{name}}_tensor_ait = AITData({{name}}_data, {{name}}_shape_ait, AITemplateDtype::{{dtype}}); // TODO(supuna): dtype hardcoded
    """
)

# TODO(supuna): dtype hardcoded in AIT
INIT_AIT_DATA_OUTPUT_BODY_LINE = jinja2.Template(
    """
auto {{name}}_shape_ait = AITemplateParamShape({{name}}_shape_data, {{name}}_rank);
auto {{name}}_tensor_ait = AITData({{name}}_data, {{name}}_shape_ait, AITemplateDtype::{{dtype}}); // TODO(supuna): dtype hardcoded
    """
)

SET_AIT_CONSTANTS_BODY_LINE = jinja2.Template(
    """AITemplateModelContainerSetConstant(handle, "{{constant_name}}", &{{tensor_name}}_tensor_ait);"""
)

#TODO: proper jinja for list
SET_AIT_INPUTS_BODY = jinja2.Template(
    """AITData ait_inputs[{{num_inputs}}] = { {{inputs}} };"""
)

#TODO: proper jinja for list
SET_AIT_OUTPUTS_BODY = jinja2.Template(
    """AITData ait_outputs[{{num_outputs}}] = { {{outputs}} };"""
)

#TODO: proper jinja for list
SET_AIT_OUTPUT_SHAPES_BODY_LINE = jinja2.Template(
    """int64_t *ait_output_shapes[{{num_outputs}}] = { {{output_shapes_list}} };"""
)

GET_TYPE_BODY_FIRST_LINE = jinja2.Template(
  """if (i == 0) return {{dtype}};"""
)

GET_TYPE_BODY_LINE = jinja2.Template(
  """else if (i=={{id}}) return {{dtype}};"""
)