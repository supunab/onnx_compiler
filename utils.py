from __future__ import annotations
import numpy as np
import onnx
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE, NP_TYPE_TO_TENSOR_TYPE

# names are variable names in generated code, so shouldn't result in invalid var names
def clean_name(name: str) -> str:
    name = name.replace("::", "_") # some var names are onnx::gemm_input
    name = name.replace(".", "_") # some var names conatins dots
    if not name[0].isalpha(): # some var names only contain numbers
        name = "_" + name
    return name

def map_onnx_dtype_to_numpy(onnx_dtype: int):
    return TENSOR_TYPE_TO_NP_TYPE[onnx_dtype]

def map_np_type_to_onnx(np_type):
    np_type = np.dtype(np_type)
    return NP_TYPE_TO_TENSOR_TYPE[np_type] 

def map_type(elem_type: int) -> str:
    np_type = TENSOR_TYPE_TO_NP_TYPE[elem_type]
    # TODO: add support for types as needed!
    # https://github.com/onnx/onnx/blob/4e0b7197a015549f6773d22d174f854d7782295d/onnx/mapping.py#L13
    if np_type == np.dtype("float16"):
        return "float16"
    elif np_type == np.dtype("float32"):
        return "float"
    elif np_type == np.dtype("uint8"):
        return "uint8"
    elif np_type == np.dtype("int64"):
        return "int64"
    elif np_type == np.dtype("int32"):
        return "int32"
    else:
        raise NotImplementedError(f"type mapping for {elem_type} (np type = {np_type}) is not implemented yet")


def map_type_to_onnx_str(type) -> str:
    if type == "float16":
        return "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16"
    elif type == "float":
        return "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT"
    elif type == "uint8":
        return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8"
    elif type == "int64":
        return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64"
    elif type == "int32":
        return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32"
    else:
        raise NotImplementedError(f"type mapping for {type} is not implemented yet")

    # #TODO: numpy class to str mapping a bit ugly
    # type = str(type)
    # if type.startswith("<class"):
    #     type = type.split("'")[1]
    #     if type.startswith("numpy"):        
    #         type = type.split(".")[1]
    
    # if type == "float":
    #     return "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT"
    # elif type == "int64":
    #     return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64"
    # elif type == "float16":
    #     return "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16"
    # else:
    #     raise NotImplementedError(f"type mapping for {type} is not implemented yet")


def map_type_to_ait_str(type) -> str:
    # TODO: need to check np types
    if type == np.dtype("float16"):
        return "kHalf"
    elif type == np.dtype("float"):
        return "kFloat"
    elif type == np.dtype("int32"):
        return "kInt"
    elif type == np.dtype("int64"):
        return "kLong"
    else:
        raise NotImplementedError(f"type mapping for {type} is not implemented yet")


def to_attribute_dict(attributes: list[onnx.AttributeProto]) -> dict:
    d = {}
    for attr in attributes:
        d[attr.name] = attr
    return d

def add_input_to_node(node, init_name, pos: int):
    if len(node.input) > pos:
        node.input[pos] = init_name
    else:
        for _ in range(len(node.input), pos):
            node.input.append("")
        node.input.append(init_name)
