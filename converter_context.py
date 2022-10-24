from __future__ import annotations
import onnx
from aitemplate.frontend import Tensor

## keeps track of the processed tensors
class ConverterContext:
    def __init__(self, graph: onnx.GraphProto) -> None:
        self.tensors = {}
        self.outputs = list(map(lambda t: t.name, graph.output))

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
        return self.tensors[name]

    def get_final_output(self) -> Tensor:
        # TODO: what is there are more outputs?
        assert len(self.outputs) == 1, "Only support cases where there's only one output"
        return self.tensors[self.outputs[0]]
