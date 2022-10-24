from __future__ import annotations


# names are variable names in generated code, so shouldn't result in invalid var names
def clean_name(name: str) -> str:
    name = name.replace("::", "_") # some var names are onnx::gemm_input
    name = name.replace(".", "_") # some var names conatins dots
    if not name[0].isalpha(): # some var names only contain numbers
        name = "_" + name
    return name