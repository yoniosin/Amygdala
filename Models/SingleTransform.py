from util.AmygDataSet import GlobalAmygDataset
from util.config import LearnerMetaData
from pathlib import Path
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
import progressbar
import numpy as np
import torch


class STNet(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
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


class SingleTransform:
    def __init__(self, md: LearnerMetaData):
        ds = GlobalAmygDataset(Path('../timeseries/Data/3D'), md)
        self.batch_size = md.batch_size
        self.net = STNet(ds.get_sample_shape())
        train_ds, test_ds = random_split(ds, (ds.train_len, ds.test_len))
        self.train_dl = DataLoader(train_ds, batch_size=md.batch_size, shuffle=True)
        self.test_dl = DataLoader(test_ds, batch_size=md.batch_size, shuffle=True)
        self.optimizer = optim.Adam(self.net.parameters(), lr=2e-5, weight_decay=1e-1)
        self.loss_func = nn.MSELoss()

    def train(self, n_epochs):
        bar = progressbar.ProgressBar()
        writer = SummaryWriter()
        for i in bar(range(n_epochs)):
            train_loss = self.run_model(train=True)
            writer.add_scalar('train', np.mean(train_loss), i)
            test_loss = self.test()
            writer.add_scalar('test', np.mean(test_loss), i)

        writer.close()
        torch.save(self.net, 'last_run.pt')

    def test(self):
        with torch.no_grad():
            return self.run_model(train=False)

    def run_model(self, train: bool):
        self.net.train(train)
        dl = self.train_dl if train else self.test_dl
        return [self.run_bacth(batch, train) for batch in dl]

    def run_bacth(self, batch, train: bool):
        x, y = batch
        input_ = Variable(x, requires_grad=train)
        target = Variable(y.squeeze(), requires_grad=False)
        output = self.net(input_)

        loss = self.loss_func(output, target)
        if train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return float(loss)




