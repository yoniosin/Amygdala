from util.config import fMRILearnerConfig
from util.AmygDataSet import ScoresAmygDataset
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
from MetaData.SubjectMetaData import FibroSubjectMetaData, PTSDSubjectMetaData, HealthySubjectMetaData
from dataclasses import dataclass
import Models.Predictors as Prd



@dataclass
class DBWrapper:
    db_type: str
    data_path: Path
    meta_data_path: str
    split_path: str = 'split.json'

    def generate_meta_data(self):
        if self.db_type == 'healthy':
            meta_data_type = HealthySubjectMetaData
        elif self.db_type == 'PTSD':
            meta_data_type = PTSDSubjectMetaData
        elif self.db_type == 'Fibro':
            meta_data_type = FibroSubjectMetaData
        else:
            raise ValueError(f'Illegal value for Meta data type: {self.db_type}')

        self.meta_data = meta_data_type(self.meta_data_path, self.split_path)


def load_data_set(db_wrapper_list: Iterable[DBWrapper], load):
    ds_location = Path('data')
    if load and (ds_location / 'train.pt').exists() and (ds_location / 'test.pt').exists():
        return torch.load(ds_location / 'train.pt'), torch.load(ds_location / 'test.pt'), torch.load(
            ds_location / 'input_shape.pt')

    ds = ScoresAmygDataset([db.data_path for db in db_wrapper_list], md)
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


def prepare_run():
    run_num = json.load(open('runs/last_run.json', 'r'))['last'] + 1
    binned_run_num = json.load(open('binned_runs/last_run.json', 'r'))['last_run'] + 1

    if args.create_mapping:
        create_mapping([Path('../Amygdala/data/3D'), Path('data/PTSD'), Path('data/Fibro')])

    return fMRILearnerConfig(batch_size=10,
                             train_ratio=0.9,
                             run_num=binned_run_num if binned else run_num,
                             use_embeddings=args.embed,
                             train_windows=1
                             )


def upload_db(db_type_list):
    paths = {'healthy': (Path('../Amygdala/data/3D'), '../Amygdala/MetaData/fDemog.csv'),
             'PTSD': (Path('data/PTSD'), 'MetaData/PTSD/Clinical.csv'),
             'Fibro': (Path('data/Fibro'), 'MetaData/Fibro/Clinical.csv')}

    db_list_ = [DBWrapper(db_type, *paths[db_type]) for db_type in db_type_list]
    train_dl_, test_dl_, input_shape_ = load_data_set(db_list_, load=load_ds)
    network_params_ = {'n_subjects': len(json.load(open('mapping.json', 'r'))),
                       'input_shape': input_shape_,
                       'hidden_size': net_hidden_size}
    for db in db_list_:
        db.generate_meta_data()
    return {'train_dl': train_dl_, 'test_dl': test_dl_}, network_params_, db_list_


if __name__ == '__main__':
    binned = True
    parser = ArgumentParser()
    parser.add_argument('embed', type=str, choices=['none', 'init', 'concat'])
    parser.add_argument('-m', '--create_mapping', action='store_true')
    net_hidden_size = 16
    load_ds = False

    args = parser.parse_args()
    md = prepare_run()

    data_loaders, network_params, db_list = upload_db([
        # 'healthy',
        'PTSD',
        # 'Fibro',
    ])

    embedding_params = {'embedding_layer': torch.load('sequence_last_run.pt').rnn.initilaizer,
                        'meta_data_iter': [db.meta_data for db in db_list],
                        'run_num': md.run_num,
                        'feature': 'sex',
                        'n_outputs': 5,
                        # 'net_type': 'cnn'
                        }

    model = Prd.ReconstructiveModel(**data_loaders, **network_params, md=md)
    # model = Prd.EmbeddingClassifier(**data_loaders, **embedding_params)
    # model = Prd.EmbeddingClassifierBaseline(**data_loaders, **embedding_params)
    train_nn = True
    if train_nn:
        model.train(1000)
        if binned:
            json.dump({"last_run": md.run_num}, open('binned_runs/last_run.json', 'w'))
        else:
            json.dump({"last": md.run_num}, open('runs/last_run.json', 'w'))
    else:
        model.net = torch.load('sqeuence_last_run.pt')
