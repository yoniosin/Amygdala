import csv
import torch
import re
import numpy as np
from Classifier.Classifier import BaseRegression
from pathlib import Path
import json
from util.AmygDataSet import AmygDataSet, ScoresAmygDataset
from torch.utils.data import DataLoader
from util.config import LearnerMetaData
from typing import List
from abc import abstractmethod


class SubjectsMetaData:
    """
    Receives path to csv file containing meta-data for all subjects, and dumps json file ready to use.
    Dumps a list of subjects with full meta-data
    """
    def __init__(self, md_path: str, split_path: str):
        self.train_dict, self.test_dict = {}, {}
        self.train_data, self.test_data = None, None
        split = json.load(open(split_path, 'r'))
        self.train_list = split['train']
        self.test_list = split['test']

        with open(md_path) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for line in csv_reader:
                sub_num = self.extract_subject_num(line)
                try:
                    res_dict = self.get_correct_dict(sub_num)
                    res_dict[sub_num] = self.extract_meta_data(line)
                except ValueError:
                    # print(f'Failed to Create Subject #{sub_num}')
                    continue

        json.dump({**self.train_dict, **self.test_dict}, open('valid.json', 'w'))
        print(f'Created {self.name} DataSet with {len(self.full_dict)} records')

    @property
    def name(self): return

    def get_correct_dict(self, sub_num):
        if sub_num in self.test_list:
            return self.test_dict
        elif sub_num in self.train_list:
            return self.train_dict
        else:
            raise ValueError("Subject not in lists")

    @abstractmethod
    def extract_subject_num(self, line): pass

    @staticmethod
    @abstractmethod
    def extract_meta_data(line):
        """
        @:returns metadata from csv
        @:raise ValueError
        """
        pass

    @property
    def full_dict(self): return {**self.train_dict, **self.test_dict}


class HealthySubjectMetaData(SubjectsMetaData):
    @staticmethod
    def extract_meta_data(line):
        """Receives meta-data line of a subject, and returns needed values"""
        past = line['PreviousExp']
        coded_past = 2 if past == '6 EFP' else 1 if past == '1 EFP' else 0
        sex = 0 if line['Sex'] == 'M' else 1
        age = int(re.search(r'(\d{3})', line['Age']).group(1))

        return coded_past, sex, age, float(line['TAS1']), float(line['STAI_S1'])


class PTSDSubjectMetaData(SubjectsMetaData):
    @property
    def name(self): return 'PTSD'

    @staticmethod
    def extract_meta_data(line):
        if any(map(lambda t: t == '#N/A', line.values())) or line['task'] == 'Practice' or line['session'][-1] == 1:
            raise ValueError("practice run or not final session")
        return line

    def extract_subject_num(self, line):
        return str(int(line['subject']))


class FibroSubjectMetaData(SubjectsMetaData):
    @property
    def name(self): return 'Fibro'

    def extract_subject_num(self, line): return str(int(line['Subject number']))

    def extract_meta_data(self, line):
        if any(map(lambda i: i == '', line.values())):
            raise ValueError("empty value")
        return line


def load_data_set(data_location: List[Path]):
    if load and Path('test_meta_dl.pt').exists() and Path('train_meta_dl.pt').exists():
        return torch.load('train_meta_dl.pt'), torch.load('test_meta_dl.pt')

    md = LearnerMetaData(batch_size=10,
                         train_ratio=0.7,
                         run_num=100,
                         use_embeddings='init',
                         )

    ds = ScoresAmygDataset(data_location, md)
    train_ds, test_ds = ds.train_test_split()
    train_dl_ = DataLoader(train_ds, batch_size=10, shuffle=True)
    torch.save(train_dl_, 'train_meta_dl.pt')
    test_dl_ = DataLoader(test_ds, batch_size=10, shuffle=True)
    torch.save(test_dl_, 'test_meta_dl.pt')

    return train_dl_, test_dl_


if __name__ == '__main__':
    fibro_md = FibroSubjectMetaData('Fibro/Clinical.csv', '../split.json')
    ptsd_md = PTSDSubjectMetaData('PTSD/Clinical.csv', '../split.json')

    load = False
    train_dl, test_dl = load_data_set([
                                        Path('../data/PTSD'),
                                        Path('../data/Fibro'),
                                        # Path('../../data/3D'),
                                        ])
    net = torch.load('../sequence_last_run.pt')
    all_features = ['STAI', 'TAS']
    for feature in ['STAI', 'TAS']:
        emb_reg = BaseRegression(train_dl, test_dl, net.rnn.initilaizer, (fibro_md, ptsd_md), 1)
        print(f'Results for {feature} Prediction:')
        var = emb_reg.dummy_prediction_error(feature)
        print(f'    dummy = {var}')
        auxilary_features = list(set(all_features) - {feature})
        svr_loss = emb_reg.svr_prediction_error(feature, auxilary_features)
        print(f'    svr = {svr_loss}')

        # emb_reg.fit(500, feature)

    print(f'svr_loss:{svr_loss}')
