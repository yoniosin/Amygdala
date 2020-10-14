import torch
from torch import nn


class EmbeddingLSTM(nn.Module):
    """
    Regular LSTM, with the option to use subjects' embedding, in three modes:
        1) none- Regular LSTM
        2) init- cell state are initiated with the embeddings
        3) concat- embeddings are concatenated to the input
     """
    def __init__(self, spacial_size, hidden_size, n_subjects=None, embedding_size=0, num_layers=1):
        super().__init__()

        self.spacial_size = spacial_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding_layer = nn.Embedding(n_subjects, embedding_size) if embedding_size > 0 else None
        # cell_type = EmbeddingLSTMCell if embedding_size != 'concat' else ConcatEmbeddingLSTMCell
        cell_type = ConcatEmbeddingLSTMCell

        self.cells = nn.ModuleList([cell_type(hidden_size=self.hidden_size,
                                              embedding_size=embedding_size,
                                              input_size=self.spacial_size if i == 0 else self.hidden_channels[i - 1],
                                              )
                                    for i in range(num_layers)])

    def forward(self, x, subject_id, y):
        batch_size = x.shape[0]
        sequence_len = x.shape[-1]
        cells_output = []
        for cell in self.cells:
            h = cell.zero_h(batch_size)
            embedding_vec = self.get_embeddings(subject_id)
            # cell state is random unless init mode is activated
            c = embedding_vec.clone().detach() if self.embedding_layer == 'init' else h.clone().detach()

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

    def get_embeddings(self, subject_id):
        return self.embedding_layer(subject_id) if self.embedding_layer else None


class EmbeddingLSTMCell(nn.Module):
    """Vanilla LSTM cell"""
    def __init__(self, input_size, hidden_size, embedding_size=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.gates = nn.ModuleDict({g_name: self.gen_gate() for g_name in ('forget', 'input', '_update', 'out')})
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def zero_h(self, batch_size): return torch.zeros(batch_size, self.hidden_size).float()

    def gen_gate(self):
        """Creates a gate with shape [input + hidden, hidden]"""
        return nn.Linear(self.input_size + self.hidden_size + self.embedding_size, self.hidden_size)

    @staticmethod
    def rearrange_inputs(x, h_prev, _):
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

    @staticmethod
    def rearrange_inputs(x, h_prev, embed):
        """
        Concatenates input, hidden state and embedding vector to form a single matrix
        that can be passed through the cell's gates
        """
        return torch.cat((x, embed, h_prev), dim=1)


class EEGEmbedingLSTM(nn.Module):
    def __init__(self, hidden_size, embedding_size=0, n_subjects=None):
        super().__init__()

        self.n_subjects = n_subjects
        if embedding_size > 0:
            assert n_subjects > 0, "embedding size must be greater than 0 if n_subjects is provided"
            self.embedding_lut = nn.Embedding(n_subjects, embedding_size)
            # self.embedding_lut.weight.requires_grad = False
            self.arrange_inputs = self.concat_embeddings
        else:
            self.arrange_inputs = lambda x, s_id: x

        self.lstm = nn.LSTM(input_size=embedding_size + 1, hidden_size=hidden_size)

    def forward(self, x, subject_id=None):
        lstm_input = self.arrange_inputs(x, subject_id)
        out = self.lstm(lstm_input)
        return out

    def concat_embeddings(self, x, subject_id):
        seq_len, _, _ = x.shape
        embeddings = self.embedding_lut(subject_id).unsqueeze(0).expand(seq_len, -1, -1)
        return torch.cat((x, embeddings), dim=-1)


class EEGLSTM(nn.LSTM):
    def __init__(self, hidden_size, *args, **kwargs):
        super().__init__(1, hidden_size, *args, **kwargs)
