from util.config import LearnerMetaData
from util.AmygDataSet import SequenceAmygDataset
from argparse import ArgumentParser
from pathlib import Path
import torch
from torch.utils.data import random_split, DataLoader
from Models.SingleTransform import BaseModel, SequenceTransformNet
import json


def load_data_set(ds_type, load):
    ds_location = Path('data')
    if load and (ds_location / 'train_meta.pt').exists() and (ds_location / 'test_meta.pt').exists():
        return torch.load(ds_location / 'train_meta.pt'), torch.load(ds_location / 'test_meta.pt'), torch.load(
            ds_location / 'input_shape.pt')

    ds = ds_type(Path('data/3D'), md)
    train_ds, test_ds = random_split(ds, (ds.train_len, ds.test_len))
    train_list, test_list = list(map(lambda e: e.get_subjects_list, (train_ds, test_ds)))
    json.dump({'train': train_list, 'test': test_list}, open('split.json', 'w'))

    train_dl_ = DataLoader(train_ds, batch_size=md.batch_size, shuffle=True)
    torch.save(train_dl_, 'data/train_meta.pt')
    test_dl_ = DataLoader(test_ds, batch_size=md.batch_size, shuffle=True)
    torch.save(test_dl_, 'data/test_meta.pt')
    # torch.save(ds.get_sample_shape(), 'data/input_shape.pt')
    return train_dl_, test_dl_, ds.get_sample_shape()


def get_person_embedding(fc_layer):
    def create_one_hot(pesron_id):
        res = torch.zeros(87)
        res[pesron_id] = 1
        return res

    persons = torch.stack([create_one_hot(i) for i in range(87)])
    full_embedding = fc_layer(persons)
    return list(zip(persons, full_embedding))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('embed', type=str, choices=['none', 'init', 'concat'])

    args = parser.parse_args()
    run_num = json.load(open('runs/last_run.json', 'r'))['last'] + 1
    print(args.embed)
    md = LearnerMetaData(batch_size=20,
                         train_ratio=0.9,
                         run_num=run_num,
                         use_embeddings=args.embed,
                         )
    load_model = True
    train_dl, test_dl, input_shape = load_data_set(SequenceAmygDataset, load=load_model)
    model = BaseModel(input_shape, 10, md, train_dl, test_dl, net_type=SequenceTransformNet)
    train_nn = True
    if train_nn:
        model.train(50)
        json.dump({"last": run_num}, open('runs/last_run.json', 'w'))
    else:
        model.net = torch.load('sqeuence_last_run.pt')
