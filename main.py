from Models.SingleTransform import BaseModel
from util.config import LearnerMetaData
from util.AmygDataSet import SequenceAmygDataset, GlobalAmygDataset
from Models.SingleTransform import SequenceTransformNet, STNet
import json
from argparse import ArgumentParser
from pathlib import Path
import torch
from torch.utils.data import random_split, DataLoader
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import numpy as np
from functools import reduce
import pickle
from torch import nn, optim
import re
from random import random
import progressbar
from torch.autograd import Variable


def load_data_set(ds_type, md):
    ds_location = Path('data')
    if (ds_location / 'train.pt').exists() and (ds_location / 'test.pt').exists():
        return torch.load(ds_location / 'train.pt'), torch.load(ds_location / 'test.pt'), torch.load(ds_location / 'input_shape.pt')

    ds = ds_type(Path('/home/yonio/Projects/conv_gru/3d_data/3D'), md)
    train_ds, test_ds = random_split(ds, (ds.train_len, ds.test_len))
    train_dl = DataLoader(train_ds, batch_size=md.batch_size, shuffle=True)
    torch.save(train_dl, 'data/train.pt')
    test_dl = DataLoader(test_ds, batch_size=md.batch_size, shuffle=True)
    torch.save(test_dl, 'data/test.pt')
    torch.save(ds.get_sample_shape(), 'data/input_shape.pt')
    return train_dl, test_dl, ds.get_sample_shape()

def get_person_embedding(fc_layer):
    def create_one_hot(pesron_id):
        res = torch.zeros(87)
        res[pesron_id] = 1
        return res
    
    persons = torch.stack([create_one_hot(i) for i in range(87)])
    full_embedding = fc_layer(persons)
    return list(zip(persons, full_embedding))


class Regressor(nn.Module):
    def __init__(self, train_dl, test_dl, embedding_layer):
        super().__init__()
        self._embed = embedding_layer
        self.regressor = nn.Linear(10, 2)
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.optimizer = optim.Adam(self.regressor.parameters(), lr=2e0, weight_decay=0)
        self.loss_func = nn.MSELoss()
    
    def train(self, n_epochs):
        bar = progressbar.ProgressBar()
        writer = SummaryWriter()
        for i in bar(range(n_epochs)):
            train_loss = self.run_model(train=True)
            writer.add_scalar('train', np.mean([x[0] for x in train_loss]), i)
            test_loss = self.test()
            writer.add_scalar('test', np.mean([x[0] for x in test_loss]), i)
    
    def run_model(self, train: bool):
        dl = self.train_dl if train else self.test_dl
        return [self.run_batch(batch, train) for batch in dl]

    def run_batch(self, batch, train: bool):
        subject_num, x, y, subject_score, subject_one_hot = batch
        embeddings = self._embed(subject_one_hot)
        input_ = Variable(embeddings, requires_grad=train)
        target = Variable(torch.stack((subject_score), dim=-1), requires_grad=False)
        output = self.regressor(input_)

        loss = self.loss_func(output, target)
        transition_loss = 0
        if train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return float(loss), float(transition_loss)

    def test(self):
        with torch.no_grad():
            return self.run_model(train=False)


    


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', action='store_true')

    args = parser.parse_args()
    # run_num = json.load(open('runs/last_run.json', 'r'))['last'] + 1
    md = LearnerMetaData(batch_size=20,
                         train_ratio=0.9,
                         run_num=0,
                         allow_transition=args.t,
                         )
    train_dl, test_dl, input_shape = load_data_set(SequenceAmygDataset, md)
    # model = BaseModel(reduce(lambda a, b: a*b, input_shape[:-1]), [8], md, train_dl, test_dl, net_type=SequenceTransformNet)
    # model.train(50)
    # json.dump({"last": run_num}, open('runs/last_run.json', 'w'))
    # embeddings = get_person_embedding(model.net.rnn.initilaizer)
    # torch.save(embeddings, f'embeddings_{run_num}.pt')

    initilizer = torch.load('sqeuence_last_run.pt').rnn.initilaizer
    regressor = Regressor(train_dl, test_dl, initilizer)
    regressor.train(50)

    

