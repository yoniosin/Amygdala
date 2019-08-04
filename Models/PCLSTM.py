import torch
from torch import nn
from torch.autograd import Variable

class PCLSTMCell(nn.Module):
    def __init__(self,  input_channels,  hidden_size, cell_size):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        def create_gate(requires_grad): 
            layer = nn.Linear(self.input_channels * 2 + self.hidden_size, self.hidden_size)
            for key in layer._parameters:
                layer._parameters[key].requires_grad=requires_grad
            return layer

        def create_gates_dict(requires_grad=True):
            return nn.ModuleDict({'forget_gate': create_gate(requires_grad),
                                  'input_gate': create_gate(requires_grad),
                                  'update_gate': nn.Linear(800, 10),
                                  'output_gate': create_gate(requires_grad)})
        self.gates = nn.ModuleDict({'regular': create_gates_dict(), 'transition': create_gates_dict()})
        self.initial_hidden = None
    
    @property
    def regular_gates(self): return self.gates['regular']
    
    @property
    def transition_gates(self): return self.gates['transition']


    def forward(self, x, h_prev, c_prev, transition=False):
        if self.initial_hidden is None:
            raise AttributeError('Initial state not initialized')

        stacked_inputs = torch.cat((x, h_prev), dim=-1)
        gates_dict = self.transition_gates if transition else self.regular_gates
        f_t = nn.Sigmoid()(gates_dict['forget_gate'](stacked_inputs))
        i_t = nn.Sigmoid()(self.regular_gates['input_gate'](stacked_inputs))
        c_opt = nn.Tanh()(self.regular_gates['update_gate'](stacked_inputs))
        o_t = nn.Sigmoid()(self.regular_gates['output_gate'](stacked_inputs))

        c_t = c_prev * f_t + i_t * c_opt
        h_t = o_t * nn.Tanh()(c_t)

        return h_t, c_t

class PCLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels,transition_phases, allow_transition=False, num_layers=1):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.transition_phases = transition_phases
        self.allow_transition = allow_transition

        self.cells = nn.ModuleList([PCLSTMCell(hidden_size=self.hidden_channels[i], cell_size=10,
                                    input_channels=self.input_channels if i==0 else self.hidden_channels[i-1])
                                    for i in range(num_layers)])

        self.initilaizer = nn.Linear(87, self.hidden_channels[0])

    def forward(self, x, y, h_0=None):
        in_shape = x.shape
        sequence_len = in_shape[-1]
        self.init_hidden(torch.zeros(20, 87), h_0, input_shape=in_shape[:-2])
        cells_output = []
        for cell in self.cells:
            h, c = cell.initial_hidden
            inner_cell_out = []
            for t in range(sequence_len):
                x_i = x[..., t]
                y_prev = y[..., t - 1] if t > 0 else torch.zeros(x_i.shape)
                h = torch.cat((h, y_prev), dim=-1)
                use_transition_gate = self.allow_transition and t in self.transition_phases
                h, c = cell(x_i, h, c, use_transition_gate)
                inner_cell_out.append(h)

            cells_output.append(torch.stack(inner_cell_out, dim=-1))

        all_cells_output = torch.stack(cells_output, dim=1)
        return cells_output[-1], c


    def init_hidden(self, subject_id, h_0, input_shape):
        if h_0:
            for cell, h_0_i in zip(self.cells, h_0):
                cell.initial_hidden = h_0_i
                return
        else:
            zero_h_state = lambda: Variable(torch.zeros(*input_shape, self.hidden_channels[0]))
            zero_c_state = lambda id: self.initilaizer(id)
            for i, cell in enumerate(self.cells):
                cell.initial_hidden = (zero_h_state(), zero_c_state(subject_id)) #  2 for h, c
#        else:
#            zero_h_state = lambda j: Variable(torch.zeros(*input_shape, self.hidden_channels[j]))
#            init_c_state = lambda id: self.initilizer(subject_id)
#            for i, cell in enumerate(self.cells):
#                cell.initial_hidden = (zero_h_state(self.hidden_channels[0]), init_c_state(id))



