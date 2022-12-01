"""
Just trying to see if AIT mem efficient attention can support past_states and unidirectional attention
"""


## currently trying memory efficient attention
if __name__ == "__main__":
    from common import *
    from aitemplate.frontend import Tensor, nn
    from aitemplate.compiler import ops, compile_model
    from aitemplate.testing import detect_target
    import torch

    # default dtype is float16
    input_hidden_states = Tensor([batch_size, curr_seq_len, hidden_size], is_input=True, name="input_hidden_states")

    # for mem_eff_attn we need to permute (for flash attention we don't)
    qkv = nn.Linear(
                    hidden_size,
                    3 * hidden_size,
                    specialization="permute",
                    shape=(curr_seq_len, 3, num_heads),
                )
    qkv.weight.tensor()._attrs["name"] = "qkv_weight"
    qkv.bias.tensor()._attrs["name"] = "qkv_bias"
    
    # TODO: ignoring the attention mask for now
    # past0 in ONNX graph contains past_keys and values
    past_keys = Tensor([batch_size, num_heads, prev_seq_len, hidden_size // num_heads], is_input=True, name="past_keys")
    past_values = Tensor([batch_size, num_heads, prev_seq_len, hidden_size // num_heads], is_input=True, name="past_values")

    # do qkv proj on input
    input_hidden_states_reshaped = ops.reshape()(input_hidden_states, [-1, hidden_size])
    qkv_out = qkv(input_hidden_states_reshaped) # 3, bs, num_heads, sl, hidden//num_heads (figured by looking at the code)


    # q, k, v for current input
    (q, k, v) = ops.split()(qkv_out, 1, dim=0) # each having shape: 1, bs, num_heads, sl, hidden // num_heads
    k_reshaped = ops.reshape()(k, [batch_size, num_heads, curr_seq_len, hidden_size // num_heads])
    q_reshaped = ops.reshape()(q, [batch_size, num_heads, curr_seq_len, hidden_size // num_heads])
    v_reshaped = ops.reshape()(v, [batch_size, num_heads, curr_seq_len, hidden_size // num_heads])

    # concatanate old keys to k
    full_k = ops.concatenate()([past_keys, k_reshaped], dim=2)
    full_v = ops.concatenate()([past_values, v_reshaped], dim=2)
    attention_out = ops.mem_eff_attention(causal=True)(q_reshaped, full_k, full_v) # bs, sl, num_heads, hidden // num_heads
    attention_out_reshaped = ops.reshape()(attention_out, [batch_size, curr_seq_len, hidden_size])

    attention_out_reshaped._attrs["is_output"] = True
    attention_out_reshaped._attrs["name"] = "attn_out"

    # product present0 (concatanate past keys and past values with new key and value)
    full_k_reshaped = ops.reshape()(full_k, [1, batch_size, num_heads, prev_seq_len + curr_seq_len, hidden_size // num_heads])
    full_v_reshaped = ops.reshape()(full_v, [1, batch_size, num_heads, prev_seq_len + curr_seq_len, hidden_size // num_heads])
    present0 = ops.concatenate()([full_k_reshaped, full_v_reshaped], dim=0)
    
    present0._attrs["is_output"] = True
    present0._attrs["name"] = "present0"

    target = detect_target()
    with compile_model([attention_out_reshaped, present0], target, "./tmp/", "mem_eff_attn") as module:
        # setup inputs
        input_tensor = torch.randn([batch_size, curr_seq_len, hidden_size]).cuda().half()
        past_keys_tensor = torch.randn([batch_size, num_heads, prev_seq_len, hidden_size // num_heads]).cuda().half()
        past_values_tensor = torch.randn([batch_size, num_heads, prev_seq_len, hidden_size // num_heads]).cuda().half()

        # storage for output
        attention_out_tensor = torch.empty([batch_size, curr_seq_len, hidden_size]).cuda().half()
        present0 = torch.empty([2, batch_size, num_heads, prev_seq_len + curr_seq_len, hidden_size // num_heads]).cuda().half()

        # linear weights and bias
        qkv_weight_tensor = torch.randn([hidden_size, 3*hidden_size]).cuda().half()
        qkv_bias_tensor = torch.randn([3*hidden_size]).cuda().half()

        module.set_constant_with_tensor("qkv_weight", qkv_weight_tensor)
        module.set_constant_with_tensor("qkv_bias", qkv_bias_tensor)

        inputs = {
            "input_hidden_states": input_tensor,
            "past_keys": past_keys_tensor,
            "past_values": past_values_tensor
        }
        outputs = {
            "attn_out": attention_out_tensor,
            "present0": present0
        }

        module.run_with_tensors(inputs, outputs, sync=True)
        print(attention_out_tensor.cpu().numpy())
        print(present0.cpu().numpy())

