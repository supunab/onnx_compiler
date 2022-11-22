from __future__ import annotations
import os

from torch import inference_mode

from templates.custom_op import *
from templates.makefile import *
from converter_context import ConverterContext

import onnx
from onnx import helper, save

from utils import clean_name, map_type, map_type_to_onnx_str, map_type_to_ait_str

def generate_makefile(folder: str, onnx_header_path: str, ait_path: str):
    """
    Generate Makefile to compile the generated AIT sources and ORT custom op header into a shared object
    """
    # find the .cu files
    cu_files = []
    for f in os.listdir(folder):
        if f.endswith(".cu"):
            cu_files.append(f[:-3])

    obj_files = " ".join(list(map(lambda f: f + ".obj", cu_files)))
    
    with open(os.path.join(folder, "Makefile"), "w") as f:
        f.write(
            MAKEFILE_TEMPLATE.render(onnx_header_path=onnx_header_path, ait_path=ait_path, obj_files=obj_files)
        )
    
# generate the .cu and .h file required for the custom op
def generate(context: ConverterContext, folder: str, output_shape: dict = {}, inputs_order: list[int] = None, run_make = True, onnx_header_path: str = "/work/onnxruntime/include/", ait_path: str = "/work/AITemplate/"):
    """
    output_shape -> explicitly provide the output_shape when onnx graph is incapable of inferencing
    inputs_order -> HACK! this is to assign the order in which we pass the ait input tensors to ait. Currently, has to manually
                    look at the the model-generated.h param[0], param[1], .. shapes and make sure to pass the correct input (based on the shape)
    """
    outputs = context.get_all_outputs()
    constants = context.get_constants()
    inputs = context.get_inputs()

    # fill out the templates with proper variable names
    get_input_ort_values_body = []
    get_ort_value_info_body = []
    get_input_shapes_body = []
    get_input_data_ptr_body = []
    init_ait_data_body = []
    get_input_type_body = []

    i = 0
    for input in (inputs + constants):
        name = input._attrs["name"]
        dtype = input._attrs["dtype"]
        get_input_ort_values_body.append(
            GET_INPUT_ORT_VALUES_BODY_LINE.render(name = name, input_id = i)
        )
        get_ort_value_info_body.append(
            GET_ORT_VALUE_INFO_BODY_LINE.render(name = name)
        )
        get_input_shapes_body.append(
            GET_INPUT_SHAPES_BODY_LINE.render(name = name)
        )
        get_input_data_ptr_body.append(
            GET_INPUT_DATA_PTR_BODY_LINE.render(name = name)
        )
        init_ait_data_body.append(
            INIT_AIT_DATA_INPUT_BODY_LINE.render(name = name, dtype=map_type_to_ait_str(dtype))
        )
        if i==0:
            get_input_type_body.append(
                GET_TYPE_BODY_FIRST_LINE.render(dtype = map_type_to_onnx_str(dtype))
            )
        else:
            get_input_type_body.append(
                GET_TYPE_BODY_LINE.render(id = i, dtype = map_type_to_onnx_str(dtype))
            )
        i += 1

    get_input_ort_values_body = "\n".join(get_input_ort_values_body)
    get_ort_value_info_body = "\n".join(get_ort_value_info_body)
    get_input_shapes_body = "\n".join(get_input_shapes_body)
    get_input_data_ptr_body = "\n".join(get_input_data_ptr_body)
    get_input_type_body = "\n".join(get_input_type_body)

    i = 0
    output_shapes_body = []
    get_output_ort_values_body = []
    get_output_data_ptr_body = []
    output_shapes_list = []
    get_output_type_body = []
    for output in outputs:
        name = output._attrs["name"]
        dtype = output._attrs["dtype"]
        # TODO(supuna): dynamic shapes?
        # TODO(supuna): in some cases, shape inference fails, need to explicitly specify the output shape
        shape = list(map(lambda x: x.value(), output.shape()))
        shape = output_shape[name] if name in output_shape else shape
        assert not 0 in shape, f"0 usually indicates, shape inference failed, probably need to explicitly provide output shape (output={name}, shape={shape})" 

        shape_str = str(shape)[1:-1]
        rank = len(shape)
        output_shapes_body.append(
            OUTPUT_SHAPES_BODY_LINE.render(name = name, rank = rank, shape=shape_str)
        )
        get_output_ort_values_body.append(
            GET_OUTPUT_ORT_VALUES_BODY_LINE.render(name = name, output_id = i)
        )
        get_output_data_ptr_body.append(
            GET_OUTPUT_DATA_PTR_BODY_LINE.render(name = name)
        )
        init_ait_data_body.append(
            INIT_AIT_DATA_OUTPUT_BODY_LINE.render(name = name, dtype=map_type_to_ait_str(dtype))
        )
        output_shapes_list.append(
            name + "_shape_data"
        )
        if i==0:
            get_output_type_body.append(
                GET_TYPE_BODY_FIRST_LINE.render(dtype = map_type_to_onnx_str(dtype))
            )
        else:
            get_output_type_body.append(
                GET_TYPE_BODY_LINE.render(id = i, dtype = map_type_to_onnx_str(dtype))
            )
        i += 1

    set_ait_constants_body = []
    for constant in constants:
        name = constant._attrs["name"]
        set_ait_constants_body.append(
            SET_AIT_CONSTANTS_BODY_LINE.render(constant_name = name, tensor_name = name)
        )

    output_shapes_list = ",".join(output_shapes_list)
    if inputs_order != None:
        # reorder the inputs in a given order to match AIT order
        assert len(set(inputs_order)) == len(inputs)
        inputs = list(map(lambda x: x._attrs["name"] + "_tensor_ait", inputs))
        inputs_str = ",".join([inputs[i] for i in inputs_order])
    else:
        inputs_str = ",".join(map(lambda x: x._attrs["name"] + "_tensor_ait", inputs))
    outputs_str = ",".join(map(lambda x: x._attrs["name"] + "_tensor_ait", outputs))
    set_ait_inputs_body = SET_AIT_INPUTS_BODY.render(num_inputs=len(inputs), inputs=inputs_str)
    set_ait_outputs_body = SET_AIT_OUTPUTS_BODY.render(num_outputs=len(outputs), outputs=outputs_str)
    set_ait_output_shapes_body = SET_AIT_OUTPUT_SHAPES_BODY_LINE.render(num_outputs=len(outputs), output_shapes_list=output_shapes_list)
    output_shapes_body = "\n".join(output_shapes_body)
    get_output_ort_values_body = "\n".join(get_output_ort_values_body)
    get_output_data_ptr_body = "\n".join(get_output_data_ptr_body)
    init_ait_data_body = "\n".join(init_ait_data_body)
    set_ait_constants_body = "\n".join(set_ait_constants_body)
    get_output_type_body = "\n".join(get_output_type_body)
    custom_op_input_count = str(len(inputs) + len(constants))
    custom_op_output_count = str(len(outputs))
    num_inputs = str(len(inputs))
    num_outputs = str(len(outputs))

    source = CUSTOM_OP_SOURCE.render(
        get_input_ort_values_body = get_input_ort_values_body,
        get_ort_value_info_body = get_ort_value_info_body,
        get_input_shapes_body = get_input_shapes_body,
        get_input_data_ptr_body = get_input_data_ptr_body,
        output_shapes_body = output_shapes_body,
        get_output_ort_values_body = get_output_ort_values_body,
        get_output_data_ptr_body = get_output_data_ptr_body,
        init_ait_data_body = init_ait_data_body,
        set_ait_constants_body = set_ait_constants_body,
        set_ait_inputs_body = set_ait_inputs_body,
        set_ait_outputs_body = set_ait_outputs_body,
        set_ait_output_shapes_body = set_ait_output_shapes_body,
        custom_op_input_count = custom_op_input_count,
        get_input_type_body = get_input_type_body,
        custom_op_output_count = custom_op_output_count,
        get_output_type_body = get_output_type_body,
        num_inputs = num_inputs,
        num_outputs = num_outputs
    )

    with open(os.path.join(folder, "ort_ait_custom_op_library.cu"), "w") as f:
        f.write(source)

    # header file
    with open(os.path.join(folder,"ort_ait_custom_op_library.h"), "w") as f:
        f.write(CUSTOM_OP_HEADER.render())
    
    generate_makefile(folder)

    if run_make:
        # build the shared object
        # cd folder && make
        ret = os.system(f"cd {folder} && make")
        if ret != 0:
            raise Exception("Error: returned non-zero when trying to compile (i.e., make) the generated sources")


def convert_graph(old_graph: onnx.GraphProto, context: ConverterContext, model_path: str, special_inits: dict = {}):
    """
    args:
        special_inits (dict[str, TensorProto]): this is for a some special cases test/stable_diffusion/stable_diffusion_test.py
    """
    # convert the original graph into a graph containing the custom op
    # reuse the inputs and initializers from the old graph
    inputs = list(old_graph.input)
    outputs = list(old_graph.output)

    input_names = list(map(lambda x: x.name, inputs))
    output_names = list(map(lambda x: x.name, outputs))

    # context.initializers already contain the updated list of init names
    initializer_names = context.initializers

    # retrive the TensorProtos from old inits or special_inits as necessary
    init2tensor_proto = {}
    for init in old_graph.initializer:
        # let's use the cleaned name
        cleaned_name = clean_name(init.name)
        if not cleaned_name in special_inits:
            init.name = cleaned_name
            init2tensor_proto[cleaned_name] = init
    
    # create the initializer list in the same order as initializer_name
    # (don't think this is required though)
    initializers = []
    for init_name in initializer_names:
        if init_name in special_inits:
            initializers.append(special_inits[init_name])
        else:
            initializers.append(init2tensor_proto[init_name])

    # note - this ordering has to match the ordering in the generated code because we 
    #        retrieve inputs+initializers using positional indexes

    node = helper.make_node(
        op_type = "AITModelOp",
        inputs = input_names + initializer_names,
        outputs = output_names,
        name = "generated_custom_op",
        domain = "ait.customop"
    )

    graph = helper.make_graph(
        nodes = [node, ],
        name = "test-model",
        inputs = inputs,
        outputs = outputs,
        initializer = initializers
    )

    model = helper.make_model(graph, producer_name="ait-customop-generator")
    
    save(model, model_path)