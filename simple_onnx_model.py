from multiprocessing import dummy
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

if __name__=="__main__":
    input_size = 10
    hidden_size = 25
    output_size = 5
    model = SimpleModel(input_size, hidden_size, output_size)
    
    # TODO: for the moment, let's have fixed batch dim as well
    batch_size = 8
    dummy_input = torch.randn((batch_size, input_size))

    # TODO: right now exported as fp32, but ideally should be fp16
    torch.onnx.export(model, dummy_input, "simple.onnx")