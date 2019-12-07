import torch
from torch.utils.data import Dataset, random_split
from util.config import LearnerMetaData
from pathlib import Path
import pickle
import os
import json
import re
from typing import Iterable


class AmygDataSet(Dataset):
    """
    Base class to build a pytorch dataset.
    :param subjects_path - path to pre-created json files containing subject's data
    :param md - meta-data objects"""

    def __init__(self, subjects_path: Iterable[Path], md: LearnerMetaData, load=False):
        self.subjects_dict = self.load_ds() if load else self.build_ds(subjects_path, md)
        self.train_len = int(len(self) * md.train_ratio)
        self.test_len = len(self) - self.train_len

    def build_ds(self, subjects_dir_iter: Iterable[Path], md: LearnerMetaData):
        """:returns dictionary with subjects data, which is constructed using get_subject_data()"""
        def generate_subject():
            def create_one_hot(idx):
                vec = torch.zeros(one_hot_len)
                vec[idx] = 1
                return vec

            sub = pickle.load(open(str(subject_path), 'rb'))
            sub_num = str(int(re.search(r'(\d{,4})$', sub.name).group(1)))  # remove leading 0s
            if self.is_subject_valid(sub_num):
                data = sub.get_data(md.train_windows, md.min_w, scalar_result=False)
                res_list.append({'sub_num': sub_num,
                                 'one_hot': create_one_hot(mapping[sub_num]),
                                 'data': data,
                                 'input_shape': data.shape,
                                 'score': sub.get_score(md.train_windows),
                                 'type': sub.type})

        try:
            mapping = json.load(open('mapping.json', 'r'))
        except FileNotFoundError:
            mapping = json.load(open('../mapping.json', 'r'))
        one_hot_len = len(mapping)
        res_list = []
        for subjects_dir in subjects_dir_iter:
            for subject_path in subjects_dir.iterdir():
                generate_subject()

        return res_list

    def is_subject_valid(self, subject_num):
        return True

    @staticmethod
    def load_ds():
        bold_file_name = os.path.join('data', '_'.join(('3d', 'dataset.pt')))
        if os.path.isfile(bold_file_name):
            return torch.load(bold_file_name)
        else:
            raise IOError('Missing Train File')

    def __len__(self):
        return len(self.subjects_dict)

    def __getitem__(self, item):
        return self.subjects_dict[item]

    def get_sample_shape(self):
        return self.subjects_dict[0]['input_shape']

    def get_subjects_list(self):
        return list(map(lambda x: x[0].item(), self))

    def save(self):
        torch.save(self.subjects_dict, open('_'.join(('3d', 'dataset.pt')), 'wb'))

    def train_test_split(self):
        train_ds, test_ds = random_split(self, (self.train_len, self.test_len))
        train_list = [e['sub_num'] for e in train_ds]
        test_list = [e['sub_num'] for e in test_ds]
        json.dump({'train': train_list, 'test': test_list}, open('split.json', 'w'))

        return train_ds, test_ds


class ScoresAmygDataset(AmygDataSet):
    """
    kwargs should include 'subject_path' and 'md'
    """
    def __init__(self, subjects_data_path: Iterable[Path], md: LearnerMetaData, load=False):
        self.invalid = json.load(open('invalid_subjects.json', 'r'))
        super().__init__(subjects_data_path, md, load)

    def is_subject_valid(self, subject_num):
        return int(subject_num) not in self.invalid

