from util.AmygDataSet import AmygDataset, GlobalAmygDataset
from util.config import LearnerMetaData
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
import progressbar
import numpy as np
import torch
from Models.PCLSTM import PCLSTM


class STNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        input_shape = kwargs['input_shape']
        sizes = [1, 2, 4, 8, 8, 4, 2, 1]
        self.layers_n = len(sizes) - 1
        for i in range(self.layers_n):
            in_size = sizes[i]
            out_size = sizes[i+1]
            setattr(self, f'l{i}', nn.Linear(input_shape[-1] * in_size, input_shape[-1] * out_size))

    def activation(self, i):
        if i % 2 == 0:
            return nn.ReLU()
        if i < self.layers_n / 2:
            return nn.Dropout()
        return nn.Sigmoid()

    def forward(self, x):
        for i in range(self.layers_n):
            linear_i = getattr(self, f'l{i}')
            activation_i = nn.Dropout() if i % 2 == 0 else nn.ReLU()
            x = activation_i((linear_i(x)))

        return x


class SequenceTransformNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.single_transform = torch.load(open('single_run.pt', 'rb'))
        self.rnn = PCLSTM(10, [10], sub_sequence_len=14, allow_transition=kwargs.get('allow_transition', False))

    def forward(self, x, y):
        split = torch.split(x, 1, dim=1)
        xT = torch.cat([self.single_transform(x_i) for x_i in split], dim=-1)
        return self.rnn(xT.squeeze(1), y)


class BaseModel:
    def __init__(self, md: LearnerMetaData, train_dl, test_dl, input_shape, net_type, name='sqeuence'):
        self.batch_size = md.batch_size
        self.net = net_type(input_shape=input_shape, allow_transition=md.allow_transition)
        self.name = name
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.run_name = md.run_name
        self.optimizer = optim.Adam(self.net.parameters(), lr=2e0, weight_decay=0)
        self.loss_func = nn.MSELoss()

    def train(self, n_epochs):
        bar = progressbar.ProgressBar()
        writer = SummaryWriter(self.run_name)
        for i in bar(range(n_epochs)):
            train_loss = self.run_model(train=True)
            writer.add_scalar('train', np.mean(train_loss), i)
            test_loss = self.test()
            writer.add_scalar('test', np.mean(test_loss), i)

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
        x, y = batch
        ds_shape = y.shape
        y = y.view(ds_shape[0], *ds_shape[2:-1], ds_shape[1] * ds_shape[-1])
        input_ = Variable(x, requires_grad=train)
        target = Variable(y, requires_grad=False)
        output, c = self.net(input_, y)

        loss = self.loss_func(output, target)
        if train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return float(loss)
