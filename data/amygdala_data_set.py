import re
import pickle
from pathlib import Path
import json
import csv
from torch.utils.data import Dataset, random_split
from typing import Iterable, Tuple, Union
import data.subject as sub
from util.config import DataPaths
from sklearn.svm import SVR
import numpy as np
from data.subject import Criteria
import torch
from dataclasses import asdict

class AmygDataSet(Dataset):
    def __init__(self, data_paths_iter: Iterable[DataPaths], load: Path = None):
        self.invalid_subjects: list = json.load(open(
            r'C:\Users\yonio\PycharmProjects\Amygdala_new\util\invalid_subjects.json', 'rb')
        )
        self.subjects_dict = self.load_subjects(load) if load else self.create_subjects(data_paths_iter)

        # used for random access by data loaders
        self.eeg_subjects_list = list(self.subjects_dict.values())
        self.train_ds, self.test_ds = None, None

    def dump(
        self,
        dump_location=r'C:\Users\yonio\PycharmProjects\Amygdala_new\data\eeg\processed\dataset.pkl'
    ):
        with open(dump_location, 'wb') as file_:
            pickle.dump(self, file_)

    @staticmethod
    def load_subjects(data_dir: Path):
        res = {}
        for subject_path in data_dir.iterdir():
            subject: sub.EEGSubjectPTSD = pickle.load(open(subject_path, 'rb'))
            res[subject.medical_idx] = subject

        return res

    def create_subjects(self, data_paths_iter: Iterable[DataPaths]):
        subject_dict = {}
        mapping = json.load(open(r'C:\Users\yonio\PycharmProjects\Amygdala_new\mapping.json', 'r'))
        for data_dir in data_paths_iter:
            for path in Path(data_dir.eeg_dir).iterdir():
                medical_idx = str(int(re.search(r'sub-(\d+)', str(path)).group(1)))
                if system_idx := mapping.get(medical_idx):
                    subject_dict[int(medical_idx)] = sub.EEGSubjectPTSD(path, medical_idx, system_idx)
                else:
                    self.invalid_subjects.append(int(medical_idx))

        return subject_dict

    def __getitem__(self, item):
        return self.eeg_subjects_list[item].get_eeg()

    def __len__(self):
        return len(self.eeg_subjects_list)

    def train_test_split(self, train_ratio):
        train_len = int(len(self) * train_ratio)
        test_len = len(self) - train_len

        self.train_ds, self.test_ds = random_split(self, (train_len, test_len))
        return self.train_ds, self.test_ds


class CriteriaDataSet(AmygDataSet):
    def __init__(
            self,
            data_paths_iter: Iterable[DataPaths],
            load: Path = None,
            use_criteria: bool = False,
            n_outputs: int = 1
    ):
        super().__init__(data_paths_iter, load)
        self.load_criteria(data_paths_iter)
        self.criteria_subjects = [v for v in self.eeg_subjects_list if hasattr(v, 'criteria')]
        self.use_criteria = use_criteria
        self.n_outputs = n_outputs
        self.bins = self.generate_bins(n_outputs)

    @property
    def all_stats(self):
        features = set(Criteria.__annotations__.keys())
        return torch.tensor(
            [[item.get_criteria()[f] for f in features] for item in self.criteria_subjects]
        ).float()

    def generate_bins(self, n_outputs):
        features = set(Criteria.__annotations__.keys())
        bins = {}
        bins_per_side = n_outputs // 2
        mean = torch.mean(self.all_stats, dim=0)
        std = torch.std(self.all_stats, dim=0)
        for idx, feature in enumerate(features):
            bins[feature] = [mean[idx] + (j + 0.5) * std[idx] * 3
                             for j in range(-bins_per_side, bins_per_side)]

        return bins

    @property
    def var(self):
        return torch.var(self.all_stats, dim=0)

    def svr_error(self):
        def create_svr_labels(train: bool):
            ds = self.train_ds if train else self.test_ds
            ds.dataset.use_criteria = True
            aux_features = features.difference({selected_feature})
            x, y = [], []
            for item in ds.dataset.criteria_subjects:
                x.append([float(item.get_criteria()[feature]) for feature in aux_features])
                y.append(item.get_criteria()[selected_feature])

            return np.array(x), np.array(y)

        features = set(Criteria.__annotations__.keys())
        svr_error = {}
        for selected_feature in features:
            (train_x, train_y), (test_x, test_y) = [create_svr_labels(s) for s in (True, False)]
            model = SVR(gamma='auto')
            model.fit(train_x, train_y)
            y_hat = model.predict(test_x)
            svr_error[selected_feature] = np.mean((test_y - y_hat) ** 2)

        print(svr_error)

    def is_subject_valid(self, line):
        sub_num = line['subject']
        return int(sub_num) not in self.invalid_subjects and sub_num in self.subjects_dict.keys()

    def load_criteria(self, data_path_iter: Iterable[DataPaths]):
        for data_path in data_path_iter:
            meta_data_extractor = getattr(self, f'extract_meta_data_{data_path.type}')
            for line in csv.DictReader(open(data_path.criteria_dir), delimiter=','):
                if md := meta_data_extractor(line):
                    self.subjects_dict[md[0]].criteria = md[1]

    def __getitem__(self, item):
        if self.use_criteria:
            return self.criteria_get_item(item)
        else:
            return super().__getitem__(item)

    def criteria_get_item(self, item):
        res = self.criteria_subjects[item].get_criteria()
        if self.n_outputs > 1:
            for feature in sub.Criteria.__annotations__.keys():
                res[feature] = self.transform_to_bin(feature, res[feature])

        return res

    def transform_to_bin(self, feature_name, value):
        """Transforms a scalar value to a quantized value, according to the standard deviation"""
        bins = self.bins[feature_name]
        return next(
            (i for i, bin_thresh in enumerate(bins) if value < bin_thresh),
            len(bins)
        )

    def __len__(self):
        return len(self.criteria_subjects) if self.use_criteria else len(self.eeg_subjects_list)

    def extract_meta_data_ptsd(self, line) -> Union[Tuple[int, sub.Criteria], None]:
        if self.is_subject_valid_ptsd(line):
            sub_num = line['subject']
            return sub_num, sub.Criteria(
                **{
                    k: int(line[k]) for k in sub.Criteria.__annotations__.keys()
                }
            )

    def is_subject_valid_ptsd(self, line):
        return self.is_subject_valid(line) and line['task'] != 'Practice' and \
               line['session'][-1] != 1 and all((t != '#N/A' for t in line.values()))
