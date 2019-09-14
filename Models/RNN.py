import torch
from torch import nn


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, h, x):
        stacked_input = torch.cat((x, h), dim=-1)
        new_h = self.linear(stacked_input)
        return nn.Tanh()(new_h)


class RNN(nn.Module):
    def __init__(self, input_size, cells_num):
        super().__init__()
        self.cell_num = cells_num
        self.input_size = input_size
        self.rnn_cell = RNNCell(input_size, input_size)

    def forward(self, x):
        assert x.shape[0] == self.cell_num  # number of cells is equal to number of inputs

        split_x = torch.split(x, 1, dim=0)
        h_i = torch.zeros(split_x[0].shape)
        output = []

        for x_i in split_x:
            output.append(self.rnn_cell(h_i, x_i))
            h_i = output[-1]

        return torch.stack(output).squeeze().transpose(0, 1)
