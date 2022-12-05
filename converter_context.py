from __future__ import annotations
import onnx
from aitemplate.frontend import Tensor
from utils import clean_name


## a wrapper for onnx graphs that contain an indexed graphical representation
class ModelWraper:
    def __init__(self, model):
        self.model = model
        self.graph = model.graph
        consumers, outputs, producers = ModelWraper._find_consumers(model.graph)
        self.consumers = consumers
        self.outputs = outputs # not to be confused with the actual outputs of the full model (instead, it is output of each node)
        self.producers = producers
        self.removed_nodes = set()

        self.name2init = {}
        for init in model.graph.initializer:
            self.name2init[init.name] = init
    
    def is_init(self, name: str) -> bool:
        return name in self.name2init


    def _find_consumers(graph: onnx.GraphProto) -> dict:
        # note - cannot assume topological ordering
        i = 0 # to uniquely name nodes without names
        consumers = {} # consumers for a given node's given output
        outputs = {} # outputs produced by a given node
        producers = {} # producer of a particular tensor (name)
        # populate producers with inputs and inits
        for init in graph.initializer:
            if init.name == "":
                init.name = f"node_{i}"
                i += 1
            producers[init.name] = init
            consumers[init.name] = {init.name : []}
        for input in graph.input:
            if input.name == "":
                input.name = f"node_{i}"
                i += 1
            producers[input.name] = input
            consumers[input.name] = {input.name : []}

        for node in graph.node:
            if node.name == "":
                node.name = f"node_{i}"
                i +=1 
            consumers[node.name] = {}
            output_names = list(node.output)
            outputs[node.name] = output_names
            for output in output_names:
                producers[output] = node
                consumers[node.name][output] = []
            
            input_names = list(node.input)
            for input in input_names:
                if input!="": # inputs are positional so there can be empty inputs in the middle
                    producer = producers[input]
                    consumers[producer.name][input].append(node)
        return consumers, outputs, producers
            

## keeps track of the processed tensors
class ConverterContext:
    def __init__(self, graph: onnx.GraphProto, modelw: ModelWraper, attributes: dict = {}) -> None:
        self.tensors = {}
        self.outputs = list(map(lambda t: clean_name(t.name), graph.output))
        self.inputs = list(map(lambda t: clean_name(t.name), graph.input))
        self.initializers = list(map(lambda t: clean_name(t.name), graph.initializer))
        self.attributes = attributes
        self.modelw = modelw
        self.arch = "" # arch of the gpu (used in codegen)

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

    def get_final_output(self) -> list[Tensor]:
        return list(map(lambda out: self.tensors[out], self.outputs))

    def get_inputs_and_constants(self) -> list[Tensor]:
        return list(map(lambda t: self.tensors[t], self.inputs)) \
                     + list(map(lambda t: self.tensors[t], self.initializers))

    def get_all_outputs(self) -> list[Tensor]:
        return list(map(lambda t: self.tensors[t], self.outputs))

    def get_constants(self) -> list[Tensor]:
        return list(map(lambda t: self.tensors[t], self.initializers))

    def get_inputs(self) -> list[Tensor]:
        return list(map(lambda t: self.tensors[t], self.inputs))
