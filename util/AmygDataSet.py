import torch
from torch.utils.data import Dataset, random_split
from util.config import LearnerConfig, fMRILearnerConfig, EEGLearnerConfig
from pathlib import Path
import pickle
import os
import json
import re
from typing import Iterable
from abc import abstractmethod


class AmygDataSet(Dataset):
    """
    Base class to build a pytorch dataset.
    :param subjects_path - path to pre-created json files containing subject's data
    :param cfg - meta-data objects"""

    def __init__(self, subjects_path: Iterable[Path], cfg: LearnerConfig, load=False):
        self.subjects_list = self.load_ds() if load else self.build_ds(subjects_path, cfg)
        self.train_len = int(len(self) * cfg.train_ratio)
        self.test_len = len(self) - self.train_len

    def is_subject_valid(self, subject_num):
        return True

    @abstractmethod
    def build_ds(self, subjects_iter_dir, cfg: LearnerConfig):
        pass

    @staticmethod
    def load_ds():
        bold_file_name = os.path.join('data', '_'.join(('3d', 'dataset.pt')))
        if os.path.isfile(bold_file_name):
            return torch.load(bold_file_name)
        else:
            raise IOError('Missing Train File')

    def __len__(self):
        return len(self.subjects_list)

    def __getitem__(self, item):
        return self.subjects_list[item]

    def get_sample_shape(self):
        return self.subjects_list[0]['input_shape']

    def get_subjects_list(self):
        return list(map(lambda x: x[0].item(), self))

    def save(self):
        torch.save(self.subjects_list, open('_'.join(('3d', 'dataset.pt')), 'wb'))

    def train_test_split(self):
        train_ds, test_ds = random_split(self, (self.train_len, self.test_len))
        train_list = [e['medical_idx'] for e in train_ds]
        test_list = [e['medical_idx'] for e in test_ds]
        json.dump({'train': train_list, 'test': test_list}, open('split_eeg.json', 'w'))

        return train_ds, test_ds


class fMRIDataSet(AmygDataSet):
    def build_ds(self, subjects_dir_iter: Iterable[Path], md: fMRILearnerConfig):
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


class ScoresAmygDataset(fMRIDataSet):
    """
    kwargs should include 'subject_path' and 'md'
    """
    def __init__(self, subjects_data_path: Iterable[Path], md: LearnerConfig, load=False):
        self.invalid = json.load(open('invalid_subjects.json', 'r'))
        super().__init__(subjects_data_path, md, load)

    def is_subject_valid(self, subject_num):
        return int(subject_num) not in self.invalid


class EEGDataSet(AmygDataSet):
    def __init__(self, subjects_path: Iterable[Path], cfg: EEGLearnerConfig):
        super().__init__(subjects_path, cfg)
        self.data_shape = self.subjects_list[0].data_shape

    def build_ds(self, subjects_dir_iter: Iterable[Path], md=None):
        subjects_list = []
        for subjects_dir in subjects_dir_iter:
            for path in Path(subjects_dir).iterdir():
                with open(path, 'rb') as file:
                    subject = pickle.load(file)
                subjects_list.append(subject)

        return subjects_list

    def __len__(self):
        return len(self.subjects_list)

    def __getitem__(self, item):
        return self.subjects_list[item].get_data()

    def get_sample_shape(self): return self.data_shape
