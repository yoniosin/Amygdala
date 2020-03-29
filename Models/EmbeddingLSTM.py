import torch
from torch import nn


class EmbeddingLSTM(nn.Module):
    """
    Regular LSTM, with the option to use subjects' embedding, in three modes:
        1) none- Regular LSTM
        2) init- cell state are initiated with the embeddings
        3) concat- embeddings are concatenated to the input
     """
    def __init__(self, spacial_size, hidden_size, n_subjects, use_embeddings='none', num_layers=1):
        super().__init__()
        self.spacial_size = spacial_size
        self.hidden_size = hidden_size
        self.use_embeddings = use_embeddings
        cell_type = EmbeddingLSTMCell if use_embeddings != 'concat' else ConcatEmbeddingLSTMCell

        self.initilaizer = nn.Linear(n_subjects, self.hidden_size[0])
        self.cells = nn.ModuleList([cell_type(hidden_size=self.hidden_size[i],
                                              input_size=self.spacial_size if i == 0 else self.hidden_channels[i - 1],
                                              embedding_fc=self.initilaizer)
                                    for i in range(num_layers)])

    def forward(self, x, subject_id, y):
        in_shape = x.shape
        batch_size = in_shape[0]
        sequence_len = in_shape[-1]
        cells_output = []
        for cell in self.cells:
            h = cell.zero_h(batch_size)
            embedding_vec = cell.get_embeddings(subject_id)
            # cell state is random unless init mode is activated
            c = embedding_vec.clone().detach() if self.use_embeddings == 'init' else h.clone().detach()

            inner_cell_out = []
            for t in range(sequence_len):
                x_i = x[..., t]
                y_prev = y[..., t - 1] if t > 0 else torch.zeros(x_i.shape)
                # concatenate Xt, Yt-1
                x_i = torch.stack((x_i, y_prev), dim=-1)
                h, c = cell(x_i, h, c, embedding_vec)
                inner_cell_out.append(h)

            cells_output.append(torch.stack(inner_cell_out, dim=-1))

        return cells_output[-1], c


class EmbeddingLSTMCell(nn.Module):
    """Vanilla LSTM cell"""
    def __init__(self, input_size, hidden_size, embedding_fc):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_fc = embedding_fc

        self.gates = nn.ModuleDict({g_name: self.gen_gate() for g_name in ('forget', 'input', '_update', 'out')})
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def zero_h(self, batch_size): return torch.zeros(batch_size, self.hidden_size).float()
    def get_embeddings(self, subject_id): return self.embedding_fc(subject_id)

    def gen_gate(self):
        """Creates a gate with shape [input + hidden, hidden]"""
        return nn.Linear(self.input_size + self.hidden_size, self.hidden_size)

    @staticmethod
    def rearrange_inputs(x, h_prev, embed):
        """Concatenates input to hidden state to form a single matrix that can be passed through the cell's gates
        (Note: In the vanilla version, embedding vector is ignored)
        """
        return torch.cat((x.view(x.shape[0], -1), h_prev), dim=1)

    def forward(self, x, h_prev, c_prev, embedding_vec):
        """
        Calculates next cell state and hidden state by passing current input, previous states
        (and possibly embedding vector) through the cell's gates.
        :param x: the input tensor
        :param h_prev: previous hidden state
        :param c_prev: previous cell state
        :param embedding_vec: personal embedding vector. Only relevant in 'concat' mode

        :return: next cell state and hidden state
        """
        cat_input = self.rearrange_inputs(x, h_prev, embedding_vec)
        f_t = self.sigmoid(self.gates['forget'](cat_input))
        i_t = self.sigmoid(self.gates['input'](cat_input))
        c_opt = self.tanh(self.gates['_update'](cat_input))
        c_t = c_prev * f_t + i_t * c_opt

        o_t = self.sigmoid(self.gates['out'](cat_input))
        h_t = o_t * self.tanh(c_t)

        return h_t, c_t


class ConcatEmbeddingLSTMCell(EmbeddingLSTMCell):
    """Unlike vanilla LSTM cell, this cell supports concatenation of an embedding vector"""
    def gen_gate(self):
        """Creates a gate which supports the concatenation of the embedding vector to input and hidden state"""
        return nn.Linear(self.input_size + 2 * self.hidden_size, self.hidden_size)

    @staticmethod
    def rearrange_inputs(x, h_prev, embed):
        """
        Concatenates input, hidden state and embedding vector to form a single matrix
        that can be passed through the cell's gates
        """
        return torch.cat((x.view(x.shape[0], -1), embed, h_prev), dim=1)


