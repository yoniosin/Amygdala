import torch
from torch.utils.data import Dataset, ConcatDataset
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
    def __init__(self, subjects_path: Path, md: LearnerMetaData, load=False):
        self.subjects_dict = self.load_ds() if load else self.build_ds(subjects_path, md)
        self.train_len = int(len(self) * md.train_ratio)
        self.test_len = len(self) - self.train_len

    def build_ds(self, subjects_path: Path, md: LearnerMetaData):
        """:returns dictionary with subjects data, which is constructed using get_subject_data()"""
        res_dict = {}

        i = 0
        for subject_path in subjects_path.iterdir():
            sub = pickle.load(open(str(subject_path), 'rb'))
            sub_num = int(re.search(r'(\d{3})$', sub.name).group(1))
            if self.is_subject_valid(sub_num):
                res_dict[i] = (sub_num, *self.get_subject_data(sub, sub_num, md.train_windows, md.min_w))
                i += 1

        return res_dict

    def is_subject_valid(self, subject_num): return True

    @staticmethod
    def load_ds():
        bold_file_name = os.path.join('data', '_'.join(('3d', 'dataset.pt')))
        if os.path.isfile(bold_file_name):
            return torch.load(bold_file_name)
        else:
            raise IOError('Missing Train File')

    @staticmethod
    def create_one_hot(idx):
        vec = torch.zeros(105)
        vec[idx] = 1
        return vec

    def get_subject_data(self, subject, subject_num, train_windows, min_w):
        """:returns data, score, one_hot encoding"""
        return (subject.get_data(train_windows, min_w),
                subject.get_score(train_windows),
                self.create_one_hot(subject_num))

    def __len__(self): return len(self.subjects_dict)

    def __getitem__(self, item): return self.subjects_dict[item]

    def get_sample_shape(self): return self.subjects_dict[0][1].shape

    def get_subjects_list(self): return list(map(lambda x: x[0].item(), self))

    def save(self): torch.save(self.subjects_dict, open('_'.join(('3d', 'dataset.pt')), 'wb'))


class SingleLabeledAmygDataSet(AmygDataSet):
    """Add subject's label to the dataset"""
    def __init__(self, md: LearnerMetaData, subjects_path: Path, subjects_label):
        self.sub_type = subjects_label
        super().__init__(subjects_path, md)

    def get_subject_data(self, subject, subject_num, train_windows, min_w):
        """:returns label, data, score, one_hot encoding"""
        return (self.sub_type, *super().get_subject_data(subject, subject_num, train_windows, min_w))


def create_multi_labeled_amyg_dataset(md: LearnerMetaData, paths: Iterable[Iterable[Path, str]]):
    """Receives an iterable of tuples of sort (path, label) and returns a unified pytorch ConcatDataset"""
    datasets = [SingleLabeledAmygDataSet(md, *path) for path in paths]
    return ConcatDataset(datasets)


class ScoresAmygDataset(AmygDataSet):
    """
    kwargs should include 'subject_path' and 'md'
    """
    def __init__(self, subjects_path: Path, md: LearnerMetaData, load=False):
        self.valid_sub = json.load(open('MetaData/valid.json', 'r'))
        super().__init__(subjects_path, md, load)

    def is_subject_valid(self, subject_num): return str(subject_num) in self.valid_sub.keys()
