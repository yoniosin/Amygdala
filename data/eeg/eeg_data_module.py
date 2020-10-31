import pytorch_lightning as pl
from util.config import EEGLearnerConfig
from pathlib import Path
from torch.utils.data import DataLoader
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

    def __init__(self, cfg: EEGLearnerConfig = None, dataset_class=CriteriaDataSet, use_criteria=False):
        super().__init__()
        self.config = cfg or self.config
        self.train_ds, self.test_ds = None, None
        self.dataset_class = dataset_class
        self._phase = 1
        self.ds: CriteriaDataSet = None
        self.use_criteria = use_criteria

        if self.config.data.load:
            self.load_ds()
            self.ds.n_outputs = 3
            self.ds.bins = self.ds.generate_bins(3)
        else:
            self.build_ds()

    @property
    def phase(self): return self._phase

    @phase.setter
    def phase(self, new_val):
        self._phase = new_val
        if new_val == 3:
            self.ds.use_criteria = True
            self.train_ds, self.test_ds = self.ds.train_test_split(self.config.learner.train_ratio)

    def build_ds(self):
        paths_dir_iter = [getattr(self.config.data, f'{p}_paths') for p in self.config.data.db_type]

        ds = self.dataset_class(
            paths_dir_iter,
            load=Path(r'C:\Users\yonio\PycharmProjects\Amygdala_new\data\eeg\processed\PTSD'),
            use_criteria=self.use_criteria
        )

        self.train_ds, self.test_ds = ds.train_test_split(self.config.learner.train_ratio)
        ds.dump()
        self.ds = ds

    def load_ds(self):
        ds: AmygDataSet = pickle.load(
            open(r'C:\Users\yonio\PycharmProjects\Amygdala_new\data\eeg\processed\dataset.pkl', 'rb')
        )
        if self.config.data.re_split:
            ds.train_test_split(self.config.learner.train_ratio)

        self.ds = ds
        self.train_ds = ds.train_ds
        self.test_ds = ds.test_ds

    def train_dataloader(self) -> DataLoader:
        if self.phase in (1, 3):
            return DataLoader(self.train_ds, self.config.learner.batch_size, shuffle=True)
        elif self.phase == 2:
            return DataLoader(self.test_ds, self.config.learner.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, self.config.learner.batch_size) if self.phase == 3 else None

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=2, drop_last=True)
