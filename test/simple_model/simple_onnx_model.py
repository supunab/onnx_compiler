import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.dense1 = nn.Linear(input_size, hidden_size, bias = True)
        self.dense2 = nn.Linear(hidden_size, output_size, bias = True)

    def forward(self, input):
        hidden_states = self.dense1(input)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = F.softmax(hidden_states, -1)
        return hidden_states

def test_pytorch(input):
    from constants import input_size, hidden_size, output_size
    t = torch.tensor(input).cuda().half()
    torch.manual_seed(42) # to make sure saved weights are the same
    model = SimpleModel(input_size, hidden_size, output_size).cuda().half()
    model.eval()
    print(model.dense1.weight.detach().cpu().numpy())
    return model(t).detach().cpu().numpy()

if __name__=="__main__":
    from constants import input_size, hidden_size, output_size, batch_size
    torch.manual_seed(42)
    model = SimpleModel(input_size, hidden_size, output_size).cuda().half()
    dummy_input = torch.randn((batch_size, input_size)).cuda().half()
    torch.onnx.export(model, dummy_input, "./simple.onnx")