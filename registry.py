from __future__ import annotations

from attr import attributes
import onnx
from converter_context import ConverterContext
from aitemplate.compiler import ops
from aitemplate.frontend import nn
from utils import clean_name

def to_attribute_dict(attributes: list[onnx.AttributeProto]) -> dict:
    d = {}
    for attr in attributes:
        d[attr.name] = attr
    return d


def process_node(node: onnx.NodeProto, context: ConverterContext, model: OnnxModel):
    if node in model.removed_nodes:
        return

    # case-by-case logic for different node type
    op_type = node.op_type
    attributes = to_attribute_dict(list(node.attribute))
    
    if op_type == "Gemm":
        transA = True if ("transA" in attributes and attributes["transA"].i == 1) else False
        transB = True if ("transB" in attributes and attributes["transB"].i == 1) else False
        alpha = attributes["alpha"].f if "alpha" in attributes else 1.0
        beta = attributes["beta"].f if "beta" in attributes else 1.0
        assert alpha == 1.0, "non 1.00 alpha not supported in gemms yet"
        assert beta == 1.0, "non 1.00 beta not supported in gemms yet"

        A = "c" if transA else "r" # TODO: don't think column-major A is supported either
        B = "c" if transB else "r"
        bias = "_bias" if len(node.input) == 3 else ""
        op_name = f"gemm_{A}{B}r{bias}"
        op_func = getattr(ops, op_name)
        inputA = context.get_tensor(node.input[0])
        inputB = context.get_tensor(node.input[1])
        output_name = clean_name(node.output[0])

        if (len(node.input) == 3):
            inputC = context.get_tensor(node.input[2])
            output = op_func()(inputA, inputB, inputC)
            output._attrs["name"] = output_name
            context.add_tensor(output)
        elif (len(node.input) == 2):
            output = op_func()(inputA, inputB)
            output._attrs["name"] = output_name
            context.add_tensor(output)
        else:
            raise ValueError("Gemm must have either 2 or 3 inputs")

    elif op_type == "Relu":
        input = context.get_tensor(node.input[0])
        output_name = clean_name(node.output[0])
        output = ops.relu(input)
        ops.gemm_rcr_bias_fast_gelu

        output._attrs["name"] = output_name
        context.add_tensor(output)

    elif op_type == "Softmax":
        input = context.get_tensor(node.input[0])
        axis = int(attributes["axis"].i)
        output_name = clean_name(node.output[0])
        output = ops.softmax()(input, axis)
        output._attrs["name"] = output_name
        context.add_tensor(output)

    elif op_type == "Matmul":
        # TODO: make it more generalize
        a = context.get_tensor(node.input[0])
        b = context.get_tensor(node.input[1])
        op = "gemm_rrr"


        if a._rank >= 2 and b._rank == 2:
            # this is a gemm
            a_in = a if a._rank == 2 else ops.reshape()(a, [-1, a.shape[-1]])
            # TODO: verify all are row-major
            output = ops.gemm_rrr()(a_in, b)

            ops.gemm_rcr_bias_fast_gelu

            output_name = clean_name(node.output[0])
            output._attrs["name"] = output_name
            context.add_tensor(output)

        elif a._rank > 2 and b._rank > 2:
            # this is a bmm
            raise NotImplementedError(f"Matmul with A-rank:{a._rank} and B-rank:{b._rank} is not implemented yet")
        else:
            # other cases are not implemented yet either
            raise NotImplementedError(f"Matmul with A-rank:{a._rank} and B-rank:{b._rank} is not implemented yet")

    elif op_type == "FastGelu":
        ops.relu
        pass

    # TODO: need to add below for BERT
    # {'Cast', 'Tanh', 'Attention', 'SkipLayerNormalization', 'EmbedLayerNormalization', 'MatMul', 'Constant', 'FastGelu', 'Gather'}
    # EmbedLayerNormalization --> ops.bert_embeddings(...)
    # gather --> ops.dynamic_slice?

    # TODO: need to figure out matmul + activation fusion (does AIT require explicit specialization? --> kinda straightforward to implement this tho?)

    else:
        raise NotImplementedError(f"Conversion logic for {op_type} not implemented yet")