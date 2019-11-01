from util.config import LearnerMetaData
from torch import nn, optim
import progressbar
import numpy as np
import torch
from Models.EmbeddingLSTM import EmbeddingLSTM
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from abc import abstractmethod


class STNet(nn.Module):
    """This Module finds a mapping between passive windows to active windows, using a FCN"""
    def __init__(self, **kwargs):
        super().__init__()
        input_shape = kwargs['input_shape']
        sizes = [1, 2, 4, 8, 8, 4, 2, 1]
        self.layers_n = len(sizes) - 1
        for i in range(self.layers_n):
            input_size = input_shape[-1]
            in_size = sizes[i]
            out_size = sizes[i+1]
            setattr(self, f'l{i}', nn.Linear(input_size * in_size, input_size * out_size))

    def forward(self, x):
        for i in range(self.layers_n):
            linear_i = getattr(self, f'l{i}')
            activation_i = nn.Dropout() if i % 2 == 0 else nn.ReLU()
            x = activation_i((linear_i(x)))

        return x


class SequenceTransformNet(nn.Module):
    """
    Using a Single Transformation, this module maps a series of [Active, Passive]+ last Passive to the last Active
    [P P
          + P ---> A
     A A]
    """
    def __init__(self, input_shape, hidden_size, output_size, use_embeddings):
        super().__init__()
        self.single_transform = torch.load(open('single_run.pt', 'rb'))  # load preciously trained ST
        phases, _, height, width, depth, phase_len = input_shape
        seq_len = phases * phase_len
        self.out_sizes = [height, width, depth, seq_len]
        spacial_size = height * width * depth

        # create Seq2Seq NN
        # 2 factor is because we concat recent history [Xt|Yt-1]
        self.rnn = EmbeddingLSTM(2 * spacial_size, [hidden_size], use_embeddings)
        self.fc1 = nn.Linear(seq_len * hidden_size, seq_len * spacial_size)
        self.fc2 = nn.Linear(seq_len, output_size[-1])

    def forward(self, x, subject_id, y):
        split = torch.split(x, 1, dim=1)
        xT = torch.cat([self.single_transform(x_i) for x_i in split], dim=-1)
        rnn_out, c_out = self.rnn(xT.squeeze(1), subject_id, y)
        y = self.fc1(rnn_out.view(x.shape[0], -1)).view(x.shape[0], *self.out_sizes)
        # y = self.fc2(nn.ReLU()(y))
        return y.squeeze(-1), c_out


class BaseModel:
    def __init__(self,  input_shape, hidden_size, md: LearnerMetaData, train_dl, test_dl, name='sqeuence'):
        self.name = name
        self.batch_size = md.batch_size
        self.n_windows = md.train_windows + 1
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.run_name = md.run_name
        self.net = self.build_NN(input_shape, hidden_size, md.use_embeddings)
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=2e-1, weight_decay=0)

    @abstractmethod
    def build_NN(self, input_shape, hidden_size, use_embeddings): pass

    def train(self, n_epochs):
        bar = progressbar.ProgressBar()
        writer = SummaryWriter(self.run_name)
        for i in bar(range(n_epochs)):
            train_loss = self.run_model(train=True)
            writer.add_scalar('train', np.mean([x[0] for x in train_loss]), i)
            test_loss = self.test()
            writer.add_scalar('test', np.mean([x[0] for x in test_loss]), i)

        writer.close()
        torch.save(self.net, f'{self.name}_last_run.pt')

    def test(self):
        with torch.no_grad():
            return self.run_model(train=False)

    def run_model(self, train: bool):
        self.net.train(train)
        dl = self.train_dl if train else self.test_dl
        return [self.run_batch(batch, train) for batch in dl]

    def run_batch(self, batch, train: bool):
        output, target = self.calc_signals(batch, train)
        loss = self.optimizer_step(output, target, train)

        return float(loss), 0

    @abstractmethod
    def calc_signals(self, batch, train): pass

    def optimizer_step(self, output, target, train):
        loss = self.loss_func(output, target)
        if train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss


class ReconstructingModel(BaseModel):
    def calc_signals(self, batch, train):
        x = batch['data'][:, :, 0]
        y = batch['data'][:, :, 1]
        y = torch.cat([y[:, i] for i in range(self.n_windows)], dim=-1)  # concat active windows to long sequence
        input_ = Variable(x, requires_grad=train)
        target = Variable(y, requires_grad=False)
        output, c = self.net(input_, batch['one_hot'], y)

        return output, target

    def build_NN(self, input_shape, hidden_size, use_embeddings):
        return SequenceTransformNet(input_shape, hidden_size,input_shape, use_embeddings)


class ClassifyingModel(BaseModel):
    def calc_signals(self, batch, train):
        subject_num, label, data, subject_score, subject_one_hot = batch
