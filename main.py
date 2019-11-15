from util.config import LearnerMetaData
from util.AmygDataSet import AmygDataSet
from argparse import ArgumentParser
from pathlib import Path
import torch
from torch.utils.data import random_split, DataLoader
from Models import SingleTransform as t
import json
from typing import Iterable
from collections import defaultdict


def load_data_set(subjects_dir_iter: Iterable[Path], load):
    ds_location = Path('data')
    if load and (ds_location / 'train.pt').exists() and (ds_location / 'test.pt').exists():
        return torch.load(ds_location / 'train.pt'), torch.load(ds_location / 'test.pt'), torch.load(
            ds_location / 'input_shape.pt')

    ds = AmygDataSet(subjects_dir_iter, md)
    train_ds, test_ds = random_split(ds, (ds.train_len, ds.test_len))
    train_list = [e['sub_num'] for e in train_ds]
    test_list = [e['sub_num'] for e in test_ds]
    json.dump({'train': train_list, 'test': test_list}, open('split.json', 'w'))

    train_dl_ = DataLoader(train_ds, batch_size=md.batch_size, shuffle=True,
                           # collate_fn=varied_sizes_collate
                           )
    torch.save(train_dl_, 'data/train.pt')
    test_dl_ = DataLoader(test_ds, batch_size=md.batch_size, shuffle=True,
                          # collate_fn=varied_sizes_collate
                          )
    torch.save(test_dl_, 'data/test.pt')
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


def varied_sizes_collate(batch):
    res = defaultdict(list)
    for subject in batch:
        for key in subject.keys():
            res[key].append(subject[key])

    return res


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('embed', type=str, choices=['none', 'init', 'concat'])

    args = parser.parse_args()
    run_num = json.load(open('runs/last_run.json', 'r'))['last'] + 1
    print(args.embed)
    md = LearnerMetaData(batch_size=10,
                         train_ratio=0.8,
                         run_num=run_num,
                         use_embeddings=args.embed,
                         train_windows=1
                         )
    load_model = True
    healthy_dir = Path('../Amygdala/data/3D')
    ptsd_dir = Path('data/PTSD')
    fibro_dir = Path('data/Fibro')
    dir_iter = [
        ptsd_dir,
        fibro_dir,
        # healthy_dir
    ]
    train_dl, test_dl, input_shape = load_data_set(dir_iter, load=load_model)
    # model = t.ClassifyingModel(input_shape, 16, md, train_dl, test_dl, baseline=False)
    model = t.ReconstructiveModel(input_shape, 8, md, train_dl, test_dl)
    # model = t.STModel(input_shape, 10, md, train_dl, test_dl, name="single_ptsd")
    train_nn = True
    if train_nn:
        model.train(50)
        json.dump({"last": run_num}, open('runs/last_run.json', 'w'))
    else:
        model.net = torch.load('sqeuence_last_run.pt')
