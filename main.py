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
    full_embeddings = fc_layer(persons)
    reduced = TSNE().fit_transform(full_embeddings.detach().numpy())
    return reduced


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', action='store_true')

    args = parser.parse_args()
    run_num = json.load(open('runs/last_run.json', 'r'))['last'] + 1
    md = LearnerMetaData(batch_size=20,
                         train_ratio=0.9,
                         run_num=run_num,
                         allow_transition=args.t,
                         )
    train_dl, test_dl, input_shape = load_data_set(SequenceAmygDataset, md)
    model = BaseModel(reduce(lambda a, b: a*b, input_shape[:-1]), [14], md, train_dl, test_dl, net_type=SequenceTransformNet)
    model.train(50)
    json.dump({"last": run_num}, open('runs/last_run.json', 'w'))
    embeddings = get_person_embedding(model.net.rnn.initilaizer)
    embeddings.dump(f'TSNE_{run_num}.ny')
