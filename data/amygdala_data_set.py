import re
import pickle
from pathlib import Path
import json
import csv
from torch.utils.data import Dataset, random_split
from typing import Iterable, Tuple, Union
import data.subject as sub
from util.config import DataPaths


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
    ):
        super().__init__(data_paths_iter, load)
        self.load_criteria(data_paths_iter)
        self.criteria_subjects = [v for v in self.eeg_subjects_list if hasattr(v, 'criteria')]
        self.use_criteria = use_criteria

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
            return self.criteria_subjects[item].get_criteria()
        else:
            return super().__getitem__(item)

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
