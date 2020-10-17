import re
import pickle
from pathlib import Path
import json
import csv
from torch.utils.data import Dataset, random_split
from typing import Iterable, Tuple
import data.subject as sub



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
            subject: sub.EEGSubjectPTSD = pickle.load(open(subject_path, 'rb'))
            res[subject.medical_idx] = subject

        return res

    def create_subjects(self, data_dir_iter: Iterable[Path]):
        subject_dict = {}
        mapping = json.load(open('../mapping.json', 'r'))
        for data_dir in data_dir_iter:
            for path in data_dir.iterdir():
                medical_idx = str(int(re.search(r'sub-(\d+)', str(path)).group(1)))
                if system_idx := mapping.get(medical_idx):
                    subject_dict[int(medical_idx)] = sub.EEGSubjectPTSD(path, medical_idx, system_idx)
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

    def extract_meta_data(self, line) -> Tuple[int, sub.Criteria]:
        pass

    def __getitem__(self, item):
        return self.subjects_list[item].get_data(use_criteria=True)


class PTSDCriteriaDataSet(CriteriaDataSet):
    def extract_meta_data(self, line):
        sub_num = line['subject']
        if self.is_subject_valid(line):
            return sub_num, sub.Criteria(
                **{
                    k: int(line[k]) for k in sub.Criteria.__annotations__.keys()
                }
            )

    def is_subject_valid(self, line):
        return super().is_subject_valid(line) and line['task'] == 'Practice' and \
               line['session'][-1] != 1 and all((t != '#N/A' for t in line.values()))
