from __future__ import annotations

from attr import attributes
import onnx
from converter_context import ConverterContext
from aitemplate.compiler import ops
from aitemplate.frontend import nn, Tensor
from utils import clean_name, map_onnx_dtype_to_numpy, to_attribute_dict, map_type
import numpy as np

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

        if a._rank() >= 2 and b._rank() == 2:
            # this is a gemm
            a_in = a if a._rank() == 2 else ops.reshape()(a, [-1, a.shape()[-1]])
            # all MatMul inputs are row-major because there's no transA, transB attributes
            output = ops.gemm_rrr()(a_in, b)
            # reshape the output
            output = output if a._rank() == 2 else ops.reshape()(a, a.shape()[:-1] + [b.shape()[-1]])
            output_name = clean_name(node.output[0])
            output._attrs["name"] = output_name
            context.add_tensor(output)

        elif a._rank() > 2 and b._rank() > 2:
            # this is a bmm
            raise NotImplementedError(f"Matmul with A-rank:{a._rank()} and B-rank:{b._rank()} is not implemented yet")
        else:
            # other cases are not implemented yet either
            raise NotImplementedError(f"Matmul with A-rank:{a._rank()} and B-rank:{b._rank()} is not implemented yet")

    elif op_type == "gemm_rcr_fast_gelu":
        # fused gemm + bias + fast_gelu node
        a = context.get_tensor(node.input[0])
        b = context.get_tensor(node.input[1])
        bias = context.get_tensor(node.input[2])

        output = ops.gemm_rcr_bias_fast_gelu()(a, b, bias)
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
        num_heads = attributes["num_heads"].i
        seq_len = context.attributes["seq_len"]
        batch_size = context.attributes["batch_size"]
        hidden_dim = context.attributes["hidden_size"]

        # TODO: here, we are giving up on some AIT fusion (e.g., has_residual = True would fuse the add with subsequent project)
        # note - we don't use the full MHA here. Full MHA has qkv_linear + attention + linear_bias + residual add
        #        however, we are only using this for qkv_linear + attention
        mha = nn.MultiheadAttention(dim=hidden_dim, batch_size=batch_size, seq_len=seq_len, num_heads=num_heads, qkv_bias=True, has_residual=False)
        hidden_states = context.get_tensor(node.input[0])
        qkv_weight = context.get_tensor(node.input[1])
        qkv_bias = context.get_tensor(node.input[2])
        # mask = context.get_tensor(node.input[3]) # TODO: how exactly should we use mask? currently ignored

        # update the params to use tensor we created        
        mha.qkv.weight._tensor = qkv_weight
        mha.qkv.bias._tensor = qkv_bias

        intermediate = mha.qkv_proj(hidden_states)
        output_4d = mha.attention(intermediate)
        # output is 4d with (batch_size, seq_len, num_heads, hidden / num_heads)
        # convert this to 3d (as the onnx attention op) so that the shape is (batch_size, seq_len, hidden)
        output = ops.reshape()(output_4d, [output_4d.shape()[0], output_4d.shape()[1], -1])
        output_name = clean_name(node.output[0])
        output._attrs["name"] = output_name
        context.add_tensor(output)

    elif op_type == "gemm_rcr_bias_add":
        matmul_A = context.get_tensor(node.input[0])
        matmul_B = context.get_tensor(node.input[1])
        bias_weight = context.get_tensor(node.input[2])
        residual_weight = context.get_tensor(node.input[3])

        assert matmul_B._rank() == 2, f"matmul B has to be rank-2, got rank {matmul_B._rank()}"
        assert matmul_A._rank() >= 2, f"matmul A has to be rank>2, got rank {matmul_B._rank()}"
        
        matmul_A_in = matmul_A if matmul_A._rank() == 2 else ops.reshape()(matmul_A, [-1, matmul_A.shape()[-1]])
        residual_weight_in = residual_weight if residual_weight._rank() == 2 else ops.reshape()(residual_weight, [-1, residual_weight.shape()[-1]])

        output = ops.gemm_rcr_bias_add()(matmul_A_in, matmul_B, bias_weight, residual_weight_in)
        output = output if matmul_A._rank() == 2 else ops.reshape()(output, matmul_A.shape()[:-1] + [matmul_B.shape()[0]])
        output_name = clean_name(node.output[0])
        output._attrs["name"] = output_name
        context.add_tensor(output)

    elif op_type == "LayerNormalization":
        hidden_states = context.get_tensor(node.input[0])
        ln_weight = context.get_tensor(node.input[1])
        ln_bias = context.get_tensor(node.input[2])

        hidden_size = hidden_states.shape()[-1]
        epsilon = attributes["epsilon"].f
        output = ops.layernorm((hidden_size,))(x=hidden_states, gamma=ln_weight, beta=ln_bias, eps=epsilon)
        output_name = clean_name(node.output[0])
        output._attrs["name"] = output_name
        context.add_tensor(output)


    elif op_type == "EmbedLayerNormalization":
        input_ids = context.get_tensor(node.input[0])
        token_type_ids = context.get_tensor(node.input[1])
        word_embedding_weight = context.get_tensor(node.input[2])
        pos_embedding_weight = context.get_tensor(node.input[3])
        token_type_embedding_weight = context.get_tensor(node.input[4])
        ln_weight = context.get_tensor(node.input[5])
        ln_bias = context.get_tensor(node.input[6])
        # attention_mask = context.get_tensor(node.input[7]) # TODO: attention mask is not used in AIT bert_embedding?
        # pos_ids = context.get_tensor(node.input[8])
        pos_ids = context.get_tensor(node.input[7])

        epsilon = attributes["epsilon"].f

        # computes: embedding = layernorm(word_embedding + token_type_embedding + position_embedding)
        output = ops.bert_embeddings()(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=pos_ids,
            word_embeddings=word_embedding_weight,
            token_type_embeddings=token_type_embedding_weight,
            position_embeddings=pos_embedding_weight,
            gamma=ln_weight,
            beta=ln_bias,
            eps=epsilon
        )
        output_name = clean_name(node.output[0])
        output._attrs["name"] = output_name
        context.add_tensor(output)

    elif op_type == "Gather":
        """
        Gather operates a bit differnet in ONNX vs. AIT (=PyTorch). In AIT, input and indices must have the same rank.
        Output shape is equal to indices shape. 
        out[i][j][k] = input[ indices[i][j][k] ][j][k] if axis = 0,
        out[i][j][j] = input[i][ indices[i][j][k] ][k] if axis = 1, and so on

        But in ONNX, you just give a list of ints as the indices and an axis, and the output will simply pick those indices from
        the input tensor and not care about the other dims. 

        (doing this here instead of during graph transformation because we have more shape information at this point)
        """
        axis = attributes["axis"].i
        data = context.get_tensor(node.input[0])
        indices = context.get_tensor(node.input[1])
        only_one_idx = indices._rank() == 0 # to make sure that the result of the gather drops the axis dim
        if indices._rank() != data._rank():
            # need to convert indices into the AIT/PT style
            idx_init = context.modelw.name2init[indices._attrs["name"]]
            assert indices._rank() <= 1
            # read the const indices
            values = np.frombuffer(idx_init.raw_data, dtype=map_onnx_dtype_to_numpy(idx_init.data_type))
            # repeat values to create a tensor with the correct shape
            shape = list(map(lambda x: x.value(), data.shape()))
            product = 1
            for i in shape:
                product *= i
            repeat_count = product / shape[axis]

            # create a tensor with axis indices as the last dim
            new_data = np.repeat(values, repeat_count).reshape(shape[:axis] + shape[axis+1:] + [len(values)])
            # bring the axis dim into correct position
            permute_order = []
            for i in range(len(shape)):
                if i < axis:
                    permute_order.append(i)
                elif i == axis:
                    permute_order.append(len(shape) - 1)
                else:
                    permute_order.append(i - 1)
            new_data = new_data.transpose(permute_order)

            # update the initializer
            idx_init.raw_data = new_data.tobytes()
            for i in new_data.shape:
                idx_init.dims.append(i)
            
            # create a new tensor with the updated shape
            context.tensors[indices._attrs["name"]] = Tensor(shape=new_data.shape, name=indices._attrs["name"], dtype=map_type(idx_init.data_type))
            # update the reference to the new tensor
            indices = context.get_tensor(node.input[1])

        output = ops.gather()(data, axis, indices)
        output_name = clean_name(node.output[0])
        if only_one_idx:
            # need to reshape and drop the axis dim
            out_shape = data.shape()[:axis] + data.shape()[axis + 1 if axis>=0 else axis - 1:]
            output_reshaped = ops.reshape()(output, out_shape)
            output_reshaped._attrs["name"] = output_name
            context.add_tensor(output_reshaped)
        else:
            output._attrs["name"] = output_name
            context.add_tensor(output)
        
    # TODO: need to figure out matmul + activation fusion (does AIT require explicit specialization? --> kinda straightforward to implement this tho?)

    else:
        raise NotImplementedError(f"Conversion logic for {op_type} not implemented yet")