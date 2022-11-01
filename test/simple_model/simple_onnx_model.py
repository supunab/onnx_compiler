import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.benchmark as benchmark


class SimpleModel(nn.Module):
    def __init__(self, input_size, num_layers, hidden_sizes):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ParameterList()
        # first layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0], bias = True))
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1], bias = True))

    def forward(self, input):
        hidden_states = input
        for i in range(self.num_layers - 1):
            hidden_states = self.layers[i](hidden_states)
            hidden_states = F.relu(hidden_states)
        # final layers with softmax
        hidden_states = self.layers[-1](hidden_states)
        hidden_states = F.softmax(hidden_states, -1)
        return hidden_states

def _benchmark_pt(model, input):
    model(input)
    torch.cuda.synchronize()

def test_pytorch(input, bench = True):
    from constants import input_size, num_layers, hidden_sizes, warm_ups, repeats
    t = torch.tensor(input).cuda().half()
    torch.manual_seed(42) # to make sure saved weights are the same
    model = SimpleModel(input_size, num_layers, hidden_sizes).cuda().half()
    model.eval()
    if bench:
        timer = benchmark.Timer(
            stmt="_benchmark_pt(model, t)",
            setup="from simple_onnx_model import _benchmark_pt",
            globals={"model": model, "t": t}
        )
        timer.timeit(warm_ups) # warm up
        print(f"PyTorch: {timer.timeit(repeats)}")
    return model(t).detach().cpu().numpy()

if __name__=="__main__":
    from constants import input_size, num_layers, hidden_sizes, batch_size
    torch.manual_seed(42)
    model = SimpleModel(input_size, num_layers, hidden_sizes).cuda().half()
    model.eval()
    dummy_input = torch.randn((batch_size, input_size)).cuda().half()
    torch.onnx.export(model, dummy_input, "./simple.onnx")