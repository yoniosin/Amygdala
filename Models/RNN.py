import torch
from torch import nn
from itertools import chain


class RNN(nn.Module):
    def __init__(self, channels_in, hidden_size, channels_out, layers):
        super().__init__()
        self.in_linear = nn.Linear(channels_in, hidden_size)
        def gen_hidden_layer(_): return nn.Linear(hidden_size, hidden_size), nn.ReLU()
        self.hidden = nn.Sequential(*(chain(*map(gen_hidden_layer, range(layers)))))
        self.out_layer = nn.Linear(hidden_size, channels_out)

    def forward(self, x):
        x1 = nn.Tanh()(self.in_linear(x))
        x2 = self.hidden(x1)
        y = nn.Tanh()(self.out_layer(x2))

        return y.transpose(0, 1)
