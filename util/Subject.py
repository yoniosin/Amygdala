from util.config import SubjectMetaData
from itertools import chain
from abc import abstractmethod
import torch
import numpy as np
from scipy.io import loadmat
import re
import pickle
from pathlib import Path
import json
import csv
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset, random_split
from typing import Iterable, Tuple


class Window:
    def __init__(self, idx, time_slots, window_type, bold_mat):
        self.idx = idx
        self.time = time_slots
        self.window_type = window_type
        self.bold = self.gen_bold_mat(bold_mat)

    @abstractmethod
    def gen_bold_mat(self, *args): pass

    @abstractmethod
    def get_data(self, width): pass

    @property
    def mean(self): return torch.mean(self.bold)

    @property
    def std(self): return torch.std(self.bold)


class FlatWindow(Window):
    def __init__(self, idx, time_slots, window_type, bold_mat, voxels_md):
        class Voxel:
            def __init__(self, vox_coor, samples):
                self.vox_coor = vox_coor
                self.samples = samples

            def __repr__(self):
                return f"vox {self.vox_coor}"

        self.voxels = {vox: Voxel(vox, bold_mat[(*vox), time_slots]) for vox in voxels_md.amyg_vox}
        super().__init__(idx, time_slots, window_type, bold_mat)

    def gen_bold_mat(self, bold_mat):
        return torch.tensor([voxel.samples for voxel in self.voxels.values()])

    def get_data(self, width): return self.bold[:, :width]


class Window3D(Window):
    def gen_bold_mat(self, bold_mat):
        voxels_md = Subject.voxels_md
        x = bold_mat[voxels_md.h_range, :, :, :]
        x = x[:, voxels_md.w_range, :, :]
        x = x[:, :, voxels_md.d_range, :]
        return torch.tensor(x[:, :, :, self.time])

    def get_data(self, width): return self.bold[:, :, :, :width]

    def __repr__(self): return f'{self.window_type}# {self.idx}: mean={self.mean}, std={self.std}'


class PairedWindows:
    def __init__(self, watch_window, regulate_window):
        assert watch_window.idx == regulate_window.idx, f'indices mismatch: {watch_window.idx} != {regulate_window.idx}'
        self.idx = watch_window.idx
        self.watch_window: Window = watch_window
        self.regulate_window: Window = regulate_window
        self.calc_score()

    def calc_score(self):
        mean_diff = self.watch_window.mean - self.regulate_window.mean
        joint_var = 1  # torch.var(torch.cat((self.watch_window.bold, self.regulate_window.bold), dim=3))
        self.score = mean_diff / joint_var
        return self.score

    def __repr__(self):
        return f'Windows #{self.idx}, score = {self.score:.4f}'

    def get_data(self, width):
        res = torch.stack([w.get_data(width) for w in (self.watch_window, self.regulate_window)])
        return res

    @property
    def means(self): return float(self.watch_window.mean), float(self.regulate_window.mean)


class Subject:
    voxels_md = None

    def __init__(self, meta_data: SubjectMetaData, bold_mat, subject_type, window_data_type=Window3D):
        def gen_windows(wind_type):
            times_list = self.meta_data.watch_times if wind_type == 'watch' else self.meta_data.regulate_times
            return map(lambda w: window_data_type(*w, wind_type, bold_mat), enumerate(times_list))

        self.meta_data = meta_data
        self.name = meta_data.subject_name
        self.type_ = subject_type
        self.paired_windows = list(map(PairedWindows, gen_windows('watch'), gen_windows('regulate')))

    def get_data(self, train_num, width, scalar_result=True):
        if scalar_result:
            prev_data = list(chain(*[w.get_data(width) for w in self.get_windows(train_num)]))

            last_pw = self.paired_windows[train_num]
            last_data = last_pw.get_data(width)
            X = np.hstack(prev_data + last_data)
            y = last_pw.score
            return X, y
        else:
            res = torch.stack([w.get_data(width) for w in self.get_windows(train_num + 1)]).float()
            return res

    def get_single_experience(self, idx, width):
        return self.paired_windows[idx].get_data(width)

    def __repr__(self):
        grades = [pw.score for pw in self.paired_windows]
        grades_formatted = ("{:.2f}, " * len(grades)).format(*grades)
        return f'{self.type} subject #{self.name}, windows grades=[{grades_formatted}]'

    def get_windows(self, windows_num):
        return self.paired_windows[:windows_num]

    def __len__(self):
        return len(self.paired_windows)

    def get_score(self, last_window):
        return [pw.means for pw in self.paired_windows[:last_window]]
        # return self.paired_windows[0].watch_window.mean, self.paired_windows[0].regulate_window.mean

    def calc_score(self):
        for pw in self.paired_windows:
            pw.calc_score()

    @property
    def type(self):
        return 'healthy'

    def convert_to_fibro(self):
        self.__class__ = FibroSubject


class PTSDSubject(Subject):
    @property
    def type(self): return 'PTSD'


class FibroSubject(Subject):
    @property
    def type(self): return 'Fibro'


class EEGWindow(Window):
    def __init__(self, idx, bold_mat, start, length, window_type):
        super().__init__(idx, list(range(start, start + length)), window_type, bold_mat)
        self.idx = idx

    @property
    def mean(self):
        return np.mean(self.bold)

    def gen_bold_mat(self, bold_mat):
        return bold_mat[self.time]

    def get_data(self, *args):
        return torch.tensor(self.bold)


@dataclass
class Criteria:
    caps: int = None
    stai: int = None
    tas: int = None


class EEGSubjectPTSD:
    passive_duration = 20
    nf_duration = 60
    total_duration = 80
    use_clean_data = True
    data_shape = {'nf': 60, 'passive': 20}

    def __init__(self, data_path, medical_idx, system_idx):
        self.medical_idx = medical_idx
        self.system_idx = system_idx
        self.paired_windows = self.generate_windows(data_path)
        self.num_windows = None
        self.criteria = None

    def generate_windows(self, data_path):
        mat = loadmat(data_path)
        signal = mat['clean_data'] if self.use_clean_data else mat['data']
        self.num_windows = len(signal) // (self.passive_duration + self.nf_duration)

        paired_windows = []
        for w in range(self.num_windows):
            passive = EEGWindow(w, signal, w * self.total_duration, self.passive_duration, 'watch')
            nf = EEGWindow(w, signal, w * self.total_duration + self.passive_duration, self.nf_duration, 'regulate')
            paired_windows.append(PairedWindows(passive, nf))

        return paired_windows

    def get_data(self, num_windows=3, use_criteria=False):
        # num_windows = max(num_windows, self.num_windows)
        watch = torch.cat([w.watch_window.get_data() for w in self.paired_windows[:num_windows]])
        regulate = torch.cat([w.regulate_window.get_data() for w in self.paired_windows[:num_windows]])

        res = {'watch': watch, 'regulate': regulate,
               'medical_idx': self.medical_idx,
               'system_idx': self.system_idx}
        if use_criteria:
            res.update({'criteria': asdict(self.criteria)})

        return res


class AmygDataSet(Dataset):
    def __init__(self, data_dir_iter: Iterable[Path], load: Path = None):
        self.invalid_subjects: list = json.load(open(
            r'C:\Users\yonio\PycharmProjects\Amygdala_new\util\invalid_subjects.json', 'rb')
        )
        self.subjects_dict = self.load_subjects(load) if load else self.create_subjects(data_dir_iter)

        # used for random access by data loaders
        self.subjects_list = list(self.subjects_dict.values())

    def dump(self):
        with open(f'../data/eeg/processed/dataset.pkl', 'wb') as file_:
            pickle.dump(self, file_)

    @staticmethod
    def load_subjects(data_dir: Path):
        res = {}
        for subject_path in data_dir.iterdir():
            subject: EEGSubjectPTSD = pickle.load(open(subject_path, 'rb'))
            res[subject.medical_idx] = subject

        return res

    def create_subjects(self, data_dir_iter: Iterable[Path]):
        subject_dict = {}
        for data_dir in data_dir_iter:
            for path in data_dir.iterdir():
                medical_idx = str(int(re.search(r'sub-(\d+)', str(path)).group(1)))
                if system_idx := mapping.get(medical_idx):
                    subject_dict[int(medical_idx)] = EEGSubjectPTSD(path, medical_idx, system_idx)
                else:
                    self.invalid_subjects.append(int(medical_idx))

        return subject_dict

    def __getitem__(self, item):
        return self.subjects_list[item].get_data()

    def __len__(self):
        return len(self.subjects_list)

    def train_test_split(self, train_ratio):
        train_len = int(len(self) * train_ratio)
        test_len = len(self) - train_len

        return random_split(self, (train_len, test_len))


class CriteriaDataSet(AmygDataSet):
    def __init__(
            self,
            data_dir_iter: Iterable[Path],
            criteria_dir_iter: Iterable[Path],
            load: Path = None
    ):
        super().__init__(data_dir_iter, load)
        self.load_criteria(criteria_dir_iter)
        self.subjects_list = [v for v in self.subjects_list if hasattr(v, 'criteria')]

    def is_subject_valid(self, line):
        sub_num = line['subject']
        return int(sub_num) not in self.invalid_subjects and sub_num in self.subjects_dict.keys()

    def load_criteria(self, criteria_dir_iter):
        for criteria_dir in criteria_dir_iter:
            for line in csv.DictReader(open(criteria_dir), delimiter=','):
                if md := self.extract_meta_data(line):
                    self.subjects_dict[md[0]].criteria = md[1]

    def extract_meta_data(self, line) -> Tuple[int, Criteria]:
        pass

    def __getitem__(self, item):
        return self.subjects_list[item].get_data(use_criteria=True)


class PTSDCriteriaDataSet(CriteriaDataSet):
    def extract_meta_data(self, line):
        sub_num = line['subject']
        if self.is_subject_valid(line):
            return sub_num, Criteria(
                **{
                    k: int(line[k]) for k in Criteria.__annotations__.keys()
                }
            )

    def is_subject_valid(self, line):
        return super().is_subject_valid(line) and line['task'] == 'Practice' and \
               line['session'][-1] != 1 and all((t != '#N/A' for t in line.values()))


if __name__ == '__main__':
    data_dir_ = Path('../data/eeg/raw/PTSD')
    mapping = json.load(open('../mapping.json', 'r'))
    criteria_dir_ = r'C:\Users\yonio\PycharmProjects\Amygdala_new\MetaData\PTSD\Clinical.csv'

    data_set = PTSDCriteriaDataSet(data_dir_, criteria_dir_)
    data_set.dump()
