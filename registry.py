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


def process_node(node: onnx.NodeProto, context: ConverterContext):
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

        if a._rank >= 2 and b._rank == 2:
            # this is a gemm
            a_in = a if a._rank == 2 else ops.reshape()(a, [-1, a.shape[-1]])
            # all MatMul inputs are row-major because there's no transA, transB attributes
            output = ops.gemm_rrr()(a_in, b)
            output_name = clean_name(node.output[0])
            output._attrs["name"] = output_name
            context.add_tensor(output)

        elif a._rank > 2 and b._rank > 2:
            # this is a bmm
            raise NotImplementedError(f"Matmul with A-rank:{a._rank} and B-rank:{b._rank} is not implemented yet")
        else:
            # other cases are not implemented yet either
            raise NotImplementedError(f"Matmul with A-rank:{a._rank} and B-rank:{b._rank} is not implemented yet")

    elif op_type == "gemm_rcr_fast_gelu":
        # fused gemm + bias + fast_gelu node
        a = context.get_tensor(node.input[0])
        b = context.get_tensor(node.input[1])
        bias = context.get_tensor(node.input[2])

        output = ops.gemm_rcr_bias_fast_gelu(a, b, bias)
        output_name = clean_name(node.output[0])
        output._attrs["name"] = output_name
        context.add_tensor(output)

    elif op_type == "Tanh":
        input = context.get_tensor(node.input[0])
        output = ops.tanh(input)
        output_name = clean_name(node.output[0])
        output._attrs["name"] = output_name
        context.add_tensor(output)

    elif op_type == "Attention":
        seq_len = attributes["num_heads"]
        batch_size = context.attributes["batch_size"]
        hidden_dim = context.attributes["hidden_size"]

        # TODO: here, we are giving up on some AIT fusion (e.g., has_residual = True would fuse the add with subsequent project)
        # note - we don't use the full MHA here. Full MHA has qkv_linear + attention + linear_bias + residual add
        #        however, we are only using this for qkv_linear + attention
        mha = nn.MultiheadAttention(dim=hidden_dim, batch_size=batch_size, seq_len=seq_len, qkv_bias=True, has_residual=False)
        hidden_states = context.get_tensor(node.input[0])
        qkv_weight = context.get_tensor(node.input[1])
        qkv_bias = context.get_tensor(node.input[2])
        mask = context.get_tensor(node.input[3]) # TODO: how exactly should we use mask? currently ignored

        # update the params to use tensor we created        
        mha.qkv.weight._tensor = qkv_weight
        mha.qkv.bias._tensor = qkv_bias

        intermediate = mha.qkv_proj(hidden_states)
        output = mha.attention(intermediate)
        output_name = clean_name(node.output[0])
        output._attrs["name"] = output_name
        context.add_tensor(output)

    elif op_type == "SkipLayerNormalization": # TODO: probably better to unpack this and fuse matmul + add + residual --> LayerNorm
        # SkipLayerNorm is add, residual add, layer norm
        epsilon = attributes["epsilon"]
        residual = context.get_tensor(node.input[0])
        hidden_states = context.get_tensor(node.input[1])
        layer_norm_weight = context.get_tensor(node.input[2])
        layer_norm_bias = context.get_tensor(node.input[3])
        hidden_states_bias = context.get_tensor(node.input[4])

        hidden_size = context.attributes["hidden_size"]
        ln = nn.LayerNorm(normalized_shape=hidden_size, eps=epsilon)
        # update the params to use tensors we created
        ln.weight._tensor = layer_norm_weight
        ln.bias._tensor = layer_norm_bias

        # first perform the bias add
        # Note - decided to unpack the SkipLayerNorm node


    elif op_type == "EmbedLayerNormalization":
        pass

    elif op_type == "Gather":
        pass

    # TODO: need to add below for BERT
    # {'Cast', 'Tanh', 'Attention', 'SkipLayerNormalization', 'EmbedLayerNormalization', 'MatMul', 'Constant', 'FastGelu', 'Gather'}
    # EmbedLayerNormalization --> ops.bert_embeddings(...)
    # gather --> ops.dynamic_slice?

    # TODO: need to figure out matmul + activation fusion (does AIT require explicit specialization? --> kinda straightforward to implement this tho?)

    else:
        raise NotImplementedError(f"Conversion logic for {op_type} not implemented yet")