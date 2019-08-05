import torch
from torch.utils.data import Dataset
from util.config import LearnerMetaData
from pathlib import Path
import pickle
import os
import json
import re


class AmygDataset(Dataset):
    def __init__(self, subjects_path: Path, md: LearnerMetaData, load=False):
        if load:
            bold_file_name = os.path.join('data', '_'.join(('3d', 'dataset.pt')))
            if os.path.isfile(bold_file_name):
                self.data = torch.load(bold_file_name)
            else: raise IOError('Missing Train File')
        else:
            self.subjects_dict = {}
            def create_one_hot(idx):
                vec = torch.zeros(105)
                vec[idx] = 1
                return vec

            for i, subject_path in enumerate(subjects_path.iterdir()):
                subject = pickle.load(open(str(subject_path), 'rb'))
                data = subject.get_data(train_num=md.train_windows, width=md.min_w, scalar_result=False)
                subject_num = int(re.search(r'(\d{3})$', subject.name).group(1))
                subject_score = subject.get_score(md.train_windows)
                subject_one_hot = create_one_hot(subject_num)

                self.subjects_dict[i] = (subject_num, data, (subject_score), subject_one_hot)

        self.train_len = int(len(self) * md.train_ratio)
        self.test_len = len(self) - self.train_len

    def save(self):
        torch.save(self.data, open('_'.join(('3d', 'dataset.pt')), 'wb'))

    def __len__(self): return len(self.subjects_dict)

    def __getitem__(self, item):
        subject = self.data[item]
        history = subject[:-1]
        passive = subject[-1, 0]
        active = subject[-1, 1]
        return history, passive, active

    def get_sample_shape(self): return self.subjects_dict[0][1].shape


class GlobalAmygDataset(AmygDataset):
    def re_arrange(self):
        ds_shape = self.data.shape
        self.data = self.data.view(ds_shape[0] * ds_shape[1], *ds_shape[2:])

    def __getitem__(self, item):
        passive = self.data[item, 0]
        active = self.data[item, 1]
        return passive, active


class SequenceAmygDataset(AmygDataset):
    def __getitem__(self, item):
        subject = self.subjects_dict[item]
        subject_num = subject[0]
        subject_data = subject[1]
        passive = subject_data[:, 0]
        active = subject_data[:, 1]
        subject_score = subject[2]
        subject_one_hot = subject[3]

        return subject_num, passive, active, subject_score, subject_one_hot

if __name__ == '__main__':
    md_ = LearnerMetaData()
    ds = GlobalAmygDataset(Path('../../timeseries/Data/3D'), md_)
    print('data')




