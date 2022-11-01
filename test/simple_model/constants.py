# setup data
input_size = 28 * 28
# TODO: for the moment, let's have fixed batch dim as well
batch_size = 1024
num_layers = 4
hidden_sizes = [1024, 2048, 512, 10]


# for small tests
# input_size = 10
# batch_size = 2
# num_layers = 1
# hidden_sizes = [25, ]

assert num_layers == len(hidden_sizes)

# for benchmarking
warm_ups = 20
repeats = 100