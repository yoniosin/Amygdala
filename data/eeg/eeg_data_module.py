import pytorch_lightning as pl
from util.config import EEGLearnerConfig
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from data.amygdala_data_set import AmygDataSet, CriteriaDataSet
import pickle


# def generate_meta_data(self, runs_dir: Path):
#     if self.db_type == 'healthy':
#         meta_data_type = HealthySubjectMetaData
#     elif self.db_type == 'PTSD':
#         meta_data_type = PTSDSubjectMetaData
#     elif self.db_type == 'Fibro':
#         meta_data_type = FibroSubjectMetaData
#     else:
#         raise ValueError(f'Illegal value for Meta data type: {self.db_type}')
#
#     self.meta_data = meta_data_type(runs_dir/self.meta_data_path, runs_dir/self.split_path)


class EEGDataModule(pl.LightningDataModule):
    config: EEGLearnerConfig = None

    def __init__(self, cfg: EEGLearnerConfig):
        super().__init__()
        self.config = cfg
        self.train_ds, self.test_ds = None, None

    def build_ds(self):
        paths_dir_iter = [getattr(self.config.data, f'{p}_paths') for p in self.config.data.db_type]

        ds = CriteriaDataSet(
            paths_dir_iter,
            load=Path(r'C:\Users\yonio\PycharmProjects\Amygdala_new\data\eeg\processed\PTSD')
        )

        self.train_ds, self.test_ds = ds.train_test_split(self.config.learner.train_ratio)
        ds.dump()

    def setup(self, stage=None):
        if self.config.data.load:
            self.load_ds()
        else:
            self.build_ds()

    def load_ds(self):
        ds: AmygDataSet = pickle.load(
            open(r'C:\Users\yonio\PycharmProjects\Amygdala_new\data\eeg\processed\dataset.pkl', 'rb')
        )
        if self.config.data.re_split:
            ds.train_test_split(self.config.learner.train_ratio)

        self.train_ds = ds.train_ds
        self.test_ds = ds.test_ds

def test_dataloader(self):
        return DataLoader(self.test_ds)


class SecondPhaseEEGData(EEGDataModule):
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, self.config.learner.batch_size, shuffle=True)


class FirstPhaseEEGData(EEGDataModule):
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, self.config.learner.batch_size)
