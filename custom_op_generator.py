from __future__ import annotations
import os

from templates.custom_op import *
from converter_context import ConverterContext

import onnx
from onnx import helper, save

# generate the .cu and .h file required for the custom op
def generate(context: ConverterContext, folder: str):
    outputs = context.get_all_outputs()
    constants = context.get_constants()
    inputs = context.get_inputs()

    # fill out the templates with proper variable names
    get_input_ort_values_body = []
    get_ort_value_info_body = []
    get_input_shapes_body = []
    get_input_data_ptr_body = []
    init_ait_data_body = []

    i = 0
    for input in (inputs + constants):
        name = input._attrs["name"]
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
            INIT_AIT_DATA_INPUT_BODY_LINE.render(name = name)
        )
        i += 1

    get_input_ort_values_body = "\n".join(get_input_ort_values_body)
    get_ort_value_info_body = "\n".join(get_ort_value_info_body)
    get_input_shapes_body = "\n".join(get_input_shapes_body)
    get_input_data_ptr_body = "\n".join(get_input_data_ptr_body)

    i = 0
    output_shapes_body = []
    get_output_ort_values_body = []
    get_output_data_ptr_body = []
    output_shapes_list = []
    for output in outputs:
        name = output._attrs["name"]
        # TODO(supuna): dynamic shapes?
        shape = list(map(lambda x: x.value(), output.shape()))
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
            INIT_AIT_DATA_OUTPUT_BODY_LINE.render(name = name)
        )
        output_shapes_list.append(
            name + "_shape_data"
        )
        i += 1

    set_ait_constants_body = []
    for constant in constants:
        name = constant._attrs["name"]
        set_ait_constants_body.append(
            SET_AIT_CONSTANTS_BODY_LINE.render(constant_name = name, tensor_name = name)
        )

    output_shapes_list = ",".join(output_shapes_list)
    inputs_str = ",".join(map(lambda x: x._attrs["name"], inputs))
    outputs_str = ",".join(map(lambda x: x._attrs["name"], outputs))
    set_ait_inputs_body = SET_AIT_INPUTS_BODY.render(num_inputs=len(inputs), inputs=inputs_str)
    set_ait_outputs_body = SET_AIT_OUTPUTS_BODY.render(num_outputs=len(outputs), outputs=outputs_str)
    set_ait_output_shapes_body = SET_AIT_OUTPUT_SHAPES_BODY_LINE.render(num_outputs=len(outputs), output_shapes_list=output_shapes_list)
    output_shapes_body = "\n".join(output_shapes_body)
    get_output_ort_values_body = "\n".join(get_output_ort_values_body)
    get_output_data_ptr_body = "\n".join(get_output_data_ptr_body)
    init_ait_data_body = "\n".join(init_ait_data_body)
    set_ait_constants_body = "\n".join(set_ait_constants_body)
    input_count = str(len(inputs) + len(constants))
    output_count = str(len(outputs))

    # TODO: types are hardcoded to float16 atm
    get_input_type_body = "return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;"
    get_output_type_body = "return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;"

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
        input_count = input_count,
        get_input_type_body = get_input_type_body,
        output_count = output_count,
        get_output_type_body = get_output_type_body
    )

    with open(os.path.join(folder, "ort_ait_custom_op_library.cu"), "w") as f:
        f.write(source)


def convert_graph(old_graph: onnx.GraphProto, context: ConverterContext, model_path: str, special_inits: dict):
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
        if not init.name in special_inits:
            init2tensor_proto[init.name] = init
    
    # create the initializer list in the same order as initializer_name
    # (don't think this is required though)
    initializers = []
    for init_name in initializer_names:
        if init_name in special_inits:
            initializers.append(special_inits[init_name])
        else:
            initializers.append(init2tensor_proto[init.name])

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