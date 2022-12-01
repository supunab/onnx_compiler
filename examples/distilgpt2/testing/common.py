# model configs
batch_size = 1
prev_seq_len = 4
curr_seq_len = 1
hidden_size = 96 # 768
num_heads = 12

def generate_weights():
    import numpy as np
    np.random.seed(42)
    # weight in row-major (AIT might expect this in col-major, so tranpose in AIT case)
    qkv_weights = np.random.randn(hidden_size, 3 * hidden_size).astype(np.float16)
    qkv_bias = np.random.randn(3 * hidden_size).astype(np.float16)
    return (qkv_weights, qkv_bias)


def generate_inputs():
    import numpy as np
    np.random.seed(42+1)
    input = np.random.randn(batch_size, curr_seq_len, hidden_size).astype(np.float16)
    past_0 = np.random.randn(2, batch_size, num_heads, prev_seq_len, hidden_size // num_heads).astype(np.float16)
    return (input, past_0)