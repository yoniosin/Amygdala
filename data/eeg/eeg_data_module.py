import pytorch_lightning as pl
from data.AmygDataSet import EEGDataSet
from util.config import EEGLearnerConfig
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from util.Subject import PTSDCriteriaDataSet


@dataclass
class DBWrapper:
    db_type: str
    data_dir: Path
    criteria_path: str

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

    def setup(self, stage=None):
        if self.config.data.load:
            self.train_ds = torch.load(f'{self.config.learner.main_dir}/data/eeg/train.pt').dataset
            self.test_ds = torch.load(f'{self.config.learner.main_dir}/data/eeg/test.pt').dataset
        else:
            db_list = [DBWrapper(db_type, *self.config.data.paths[db_type]) for db_type in self.config.data.db_type]
            data_dir_iter = (Path(db.data_dir) for db in db_list)
            criteria_dir_iter = (Path(db.criteria_path) for db in db_list)

            ds = PTSDCriteriaDataSet(
                data_dir_iter,
                criteria_dir_iter,
                load=Path(r'C:\Users\yonio\PycharmProjects\Amygdala_new\data\eeg\processed\PTSD')
            )

            self.train_ds, self.test_ds = ds.train_test_split(self.config.learner.train_ratio)
            torch.save(self.train_ds, f'{self.config.learner.main_dir}/data/eeg/train.pt')
            torch.save(self.test_ds, f'{self.config.learner.main_dir}/data/eeg/test.pt')

    def test_dataloader(self):
        return DataLoader(self.test_ds)


class SecondPhaseEEGData(EEGDataModule):
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, self.config.learner.batch_size, shuffle=True)


class FirstPhaseEEGData(EEGDataModule):
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, self.config.learner.batch_size)

