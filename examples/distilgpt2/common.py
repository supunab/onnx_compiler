batch_size_default = 1
past_seq_len_default = 4
curr_seq_len_default = 1
num_heads_default = 12 # should match the original onnx grpah
hidden_size_default = 768 # should match the original onnx graph
vocab_size_default = 50257 # should match the original onnx graph
num_layers = 6 # should match the original onnx graph (used to generate past_0, past_1, etc. inputs)

warm_ups = 10
repeats = 30