from util.AmygDataSet import GlobalAmygDataset
from util.config import LearnerMetaData
from pathlib import Path
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
import progressbar
import numpy as np


class STNet(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_shape[-1], input_shape[-1] * 2),
                                 nn.ReLU(),
                                 # nn.Linear(input_shape[-1], input_shape[-1]),
                                 # nn.ReLU(),
                                 nn.Linear(input_shape[-1] * 2, input_shape[-1]))

    def forward(self, x):
        return self.net(x)


class SingleTransform:
    def __init__(self, md: LearnerMetaData):
        ds = GlobalAmygDataset(Path('../timeseries/Data/3D'), md)
        self.batch_size = md.batch_size
        self.net = STNet(ds.get_sample_shape())
        train_ds, test_ds = random_split(ds, (ds.train_len, ds.test_len))
        self.train_dl = DataLoader(train_ds, batch_size=md.batch_size, shuffle=True)
        self.test_dl = DataLoader(test_ds, batch_size=md.batch_size, shuffle=True)
        self.optimizer = optim.Adam(self.net.parameters(), lr=2e-3)
        self.loss_func = nn.MSELoss()

    def train(self, n_epochs):
        bar = progressbar.ProgressBar()
        writer = SummaryWriter()
        for i in bar(range(n_epochs)):
            train_loss = self.run_model(True)
            writer.add_scalar('train', np.mean(train_loss), i)
            test_loss = self.test()
            writer.add_scalar('test', np.mean(test_loss), i)

        writer.close()

    def test(self):
        return self.run_model(False)

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




