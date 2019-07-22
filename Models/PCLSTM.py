import torch
from torch import nn
from torch.autograd import Variable

class PCLSTMCell(nn.Module):
    def __init__(self,  input_channels,  hidden_channels):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.forget_gate = nn.Linear(input_channels + hidden_channels, hidden_channels)
        self.input_gate = nn.Linear(input_channels + hidden_channels, hidden_channels)
        self.update_gate = nn.Linear(input_channels + hidden_channels, hidden_channels)
        self.output_gate = nn.Linear(input_channels + hidden_channels, hidden_channels)
        self.initial_hidden = None


    def forward(self, x, h_prev, c_prev):
        if self.initial_hidden is None:
            raise AttributeError('Initial state not initialized')

        stacked_inputs = torch.cat((x, h_prev), dim=-1)
        f_t = nn.Sigmoid()(self.forget_gate(stacked_inputs))
        i_t = nn.Sigmoid()(self.input_gate(stacked_inputs))
        c_opt = nn.Tanh()(self.update_gate(stacked_inputs))
        o_t = nn.Sigmoid()(self.output_gate(stacked_inputs))

        c_t = c_prev * f_t + i_t * c_opt
        h_t = o_t * nn.Tanh()(c_t)

        return h_t, c_t

class PCLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers=1):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.cells = nn.ModuleList([PCLSTMCell(hidden_channels=self.hidden_channels[i],
                                    input_channels=self.input_channels if i==0 else self.hidden_channels[i-1])
                                    for i in range(num_layers)])

    def forward(self, x, h_0=None):
        in_shape = x.shape
        sequence_len = in_shape[-1]
        self.init_hidden(h_0, input_shape=in_shape[:-2])
        cells_output = []
        for cell in self.cells:
            h, c = cell.initial_hidden
            inner_cell_out = []
            for t in range(sequence_len):
                x_i = x[..., t]
                h, c = cell(x_i, h, c)
                inner_cell_out.append(h)

            cells_output.append(torch.stack(inner_cell_out, dim=-1))

        all_cells_output = torch.stack(cells_output, dim=1)
        return cells_output[-1], c


    def init_hidden(self, h_0, input_shape):
        if h_0:
            for cell, h_0_i in zip(self.cells, h_0):
                cell.initial_hidden = h_0_i
        else:
            zero_state = lambda j: Variable(torch.zeros(*input_shape, self.hidden_channels[j]))
            for i, cell in enumerate(self.cells):
                cell.initial_hidden = [zero_state(i) for _ in range(2)]



