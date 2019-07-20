import torch
from torch.utils.data import Dataset
from util.config import LearnerMetaData
from pathlib import Path
import pickle
import os


class AmygDataset(Dataset):
    def __init__(self, subjects_path: Path, md: LearnerMetaData, load=False):
        def get_subject_data(path):
            subject = pickle.load(open(str(path), 'rb'))
            res = subject.get_data(train_num=md.train_windows, width=md.min_w, scalar_result=False)
            return res

        if load:
            bold_file_name = os.path.join('data', '_'.join(('3d', 'dataset.pt')))
            if os.path.isfile(bold_file_name):
                self.data = torch.load(bold_file_name)
            else: raise IOError('Missing Train File')
        else:
            self.data = torch.stack([get_subject_data(p) for p in subjects_path.iterdir()])
            self.re_arrange()

        self.train_len = int(len(self) * md.train_ratio)
        self.test_len = len(self) - self.train_len

    def save(self):
        torch.save(self.data, open('_'.join(('3d', 'dataset.pt')), 'wb'))

    def re_arrange(self):
        pass

    def __len__(self): return self.data.shape[0]

    def __getitem__(self, item):
        subject = self.data[item]
        history = subject[:-1]
        passive = subject[-1, 0]
        active = subject[-1, 1]
        return history, passive, active

    def get_sample_shape(self): return self.data.shape[2:]


class GlobalAmygDataset(AmygDataset):
    def re_arrange(self):
        ds_shape = self.data.shape
        self.data = self.data.view(ds_shape[0] * ds_shape[1], *ds_shape[2:])
        # self.data = self.data[:, 0]

    def __getitem__(self, item):
        passive = self.data[item, 0]
        active = self.data[item, 1]
        return passive, active


if __name__ == '__main__':
    md_ = LearnerMetaData()
    ds = GlobalAmygDataset(Path('../../timeseries/Data/3D'), md_)
    print('data')




