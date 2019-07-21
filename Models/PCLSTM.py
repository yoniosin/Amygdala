import torch
from torch import nn

class PCLSTMCell(nn.Module):
    def __init__(self,  input_channels,  hidden_channels):
        super().__init__()
        self.forget_gate = nn.Linear(input_channels, hidden_channels)
        self.input_gate = nn.Linear(input_channels, hidden_channels)
        self.update_gate = nn.Linear(input_channels, hidden_channels)
        self.output_gate = nn.Linear(input_channels, hidden_channels)

    def forward(self, x, h, c):
        stacked_inputs = torch.cat(x, h)
        forget = nn.Sigmoid()(self.forget_gate(stacked_inputs))
        input = nn.Sigmoid()(self.input_gate(stacked_inputs))
        c_opt = nn.Tanh()(self.update_gate(stacked_inputs))

        c_new = c * forget + input * c_opt



