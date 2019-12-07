from util.config import LearnerMetaData
from util.AmygDataSet import AmygDataSet, ScoresAmygDataset
from argparse import ArgumentParser
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import json
from typing import Iterable
from collections import defaultdict
from typing import List
from itertools import chain
import re
from Classifier.Classifier import EmbeddingClassifier, EmbeddingClassifierBaseline
from MetaData.SubjectMetaData import FibroSubjectMetaData, PTSDSubjectMetaData, HealthySubjectMetaData
import Models.SingleTransform as t


def load_data_set(subjects_dir_iter: Iterable[Path], load):
    ds_location = Path('data')
    if load and (ds_location / 'train.pt').exists() and (ds_location / 'test.pt').exists():
        return torch.load(ds_location / 'train.pt'), torch.load(ds_location / 'test.pt'), torch.load(
            ds_location / 'input_shape.pt')

    ds = ScoresAmygDataset(subjects_dir_iter, md)
    train_ds, test_ds = ds.train_test_split()

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


def create_mapping(dir_iterator: List[Path]):
    joint_iter = chain(*[directory.iterdir() for directory in dir_iterator])
    subjects_dict = {int(re.search(r'(\d*).pkl$', str(path)).group(1)): i for i, path in enumerate(joint_iter)}
    json.dump(subjects_dict, open('mapping.json', 'w'))


if __name__ == '__main__':
    binned = True
    parser = ArgumentParser()
    parser.add_argument('embed', type=str, choices=['none', 'init', 'concat'])
    parser.add_argument('-m', '--create_mapping', action='store_true')

    args = parser.parse_args()
    run_num = json.load(open('runs/last_run.json', 'r'))['last'] + 1
    binned_run_num = json.load(open('binned_runs/last_run.json', 'r'))['last_run'] + 1

    if args.create_mapping:
        create_mapping([Path('../Amygdala/data/3D'), Path('data/PTSD'), Path('data/Fibro')])

    md = LearnerMetaData(batch_size=10,
                         train_ratio=0.9,
                         run_num=binned_run_num if binned else run_num,
                         use_embeddings=args.embed,
                         train_windows=1
                         )

    load_model = False
    dir_iter = [
        # Path('../Amygdala/data/3D'),
        Path('data/PTSD'),
        Path('data/Fibro'),
    ]
    n_subjects = len(json.load(open('mapping.json', 'r')))
    train_dl, test_dl, input_shape = load_data_set(dir_iter, load=load_model)
    healthy_md = HealthySubjectMetaData('../Amygdala/MetaData/fDemog.csv', 'split.json')
    fibro_md = FibroSubjectMetaData('MetaData/Fibro/Clinical.csv', 'split.json')
    ptsd_md = PTSDSubjectMetaData('MetaData/PTSD/Clinical.csv', 'split.json')
    # model = t.ClassifyingModel(n_subjects, input_shape, 16, md, train_dl, test_dl, baseline=True)
    # model = t.ReconstructiveModel(n_subjects, input_shape, 16, md, train_dl, test_dl)
    # model = t.STModel(input_shape, 10, md, train_dl, test_dl, name="single_ptsd")
    # model = BaseRegression(train_dl, test_dl, torch.load('sequence_last_run.pt').rnn.initilaizer, [ptsd_md], 1)
    model = EmbeddingClassifierBaseline(train_dl, test_dl, torch.load('sequence_last_run.pt').rnn.initilaizer,
                                        [ptsd_md, fibro_md], 'TAS', run_num=binned_run_num, n_outputs=9)
    # model = RegressiveEmbedding(train_dl, test_dl, torch.load('sequence_last_run.pt').rnn.initilaizer,
    #                             [healthy_md], 'TAS', run_num=binned_run_num)
    # model = EmbeddingClassifier(train_dl, test_dl, torch.load('sequence_last_run.pt').rnn.initilaizer,
    #                             [healthy_md], 'STAI', run_num=binned_run_num, n_outputs=5)
    train_nn = True
    if train_nn:
        model.train(500)
        if binned:
            json.dump({"last_run": binned_run_num}, open('binned_runs/last_run.json', 'w'))
        else:
            json.dump({"last": run_num}, open('runs/last_run.json', 'w'))
    else:
        model.net = torch.load('sqeuence_last_run.pt')
