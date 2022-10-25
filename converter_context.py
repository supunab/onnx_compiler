from __future__ import annotations
from unicodedata import name
import onnx
from aitemplate.frontend import Tensor
from utils import clean_name

## keeps track of the processed tensors
class ConverterContext:
    def __init__(self, graph: onnx.GraphProto) -> None:
        self.tensors = {}
        self.outputs = list(map(lambda t: clean_name(t.name), graph.output))
        self.inputs = list(map(lambda t: clean_name(t.name), graph.input))
        self.initializers = list(map(lambda t: clean_name(t.name), graph.initializer))

    def add_tensor(self, tensor: Tensor) -> None:
        name = tensor._attrs["name"]
        if not name in self.tensors:
            self.tensors[name] = tensor
            # mark as output if output tensor
            if name in self.outputs:
                tensor._attrs["is_output"] = True
        else:
            raise ValueError("Adding a tensor with a name used before")

    def get_tensor(self, name: str) -> Tensor:
        return self.tensors[clean_name(name)]

    def get_final_output(self) -> Tensor:
        # TODO: what is there are more outputs?
        assert len(self.outputs) == 1, "Only support cases where there's only one output"
        return self.tensors[self.outputs[0]]

    def get_inputs_and_constants(self) -> list[Tensor]:
        return list(map(lambda t: self.tensors[t], self.inputs)) \
                     + list(map(lambda t: self.tensors[t], self.initializers))

    def get_all_outputs(self) -> list[Tensor]:
        return list(map(lambda t: self.tensors[t], self.outputs))

    def get_constants(self) -> list[Tensor]:
        return list(map(lambda t: self.tensors[t], self.initializers))

    def get_inputs(self) -> list[Tensor]:
        return list(map(lambda t: self.tensors[t], self.inputs))
