from util.config import EEGLearnerConfig
from util.AmygDataSet import EEGDataSet
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from typing import Iterable
from dataclasses import dataclass
import Models.Networks as Net
from util.Subject import EEGSubjectPTSD, PairedWindows, EEGWindow
import hydra
from hydra.core.config_store import ConfigStore
from flatten_dict import flatten
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger


@dataclass
class DBWrapper:
    db_type: str
    data_path: Path
    meta_data_path: str
    split_path: str = 'split.json'

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


def load_data_set(db_wrapper_list: Iterable[DBWrapper], cfg):
    ds_location = Path(cfg.learner.main_dir) / 'data/eeg'
    if cfg.data.load and (ds_location / 'train.pt').exists() and (ds_location / 'test.pt').exists():
        return torch.load(ds_location / 'train.pt'), torch.load(ds_location / 'test.pt')

    ds = EEGDataSet([db.data_path for db in db_wrapper_list], cfg.learner)
    train_ds, test_ds = ds.train_test_split()

    train_dl_ = DataLoader(train_ds, batch_size=cfg.learner.batch_size, shuffle=True)
    torch.save(train_dl_, '../../../data/eeg/train.pt')
    test_dl_ = DataLoader(test_ds, batch_size=cfg.learner.batch_size, shuffle=True)
    torch.save(test_dl_, '../../../data/eeg/test.pt')
    # torch.save(ds.get_sample_shape(), 'data/input_shape.pt')
    return train_dl_, test_dl_


def upload_db(cfg):
    db_list_ = [DBWrapper(db_type, *cfg.data.paths[db_type]) for db_type in cfg.data.db_type]
    train_dl_, test_dl_ = load_data_set(db_list_, cfg)
    # for db in db_list_:
    #     db.generate_meta_data(runs_path)
    return train_dl_, test_dl_


@hydra.main(config_name='eeg')
def main(cfg: EEGLearnerConfig):
    train_dl, val_dl = upload_db(cfg)

    model = Net.EEGNetwork(
        **cfg.net,
    )
    trainer = pl.Trainer(
        logger=NeptuneLogger(
            project_name='yoniosin/amygdala',
            tags=['Embed'] if cfg.net.embedding_size > 0 else ['vanilla'],
            params=flatten(cfg, reducer='path')
        )
    )
    trainer.fit(model, train_dl, val_dataloaders=val_dl)


if __name__ == '__main__':
    cs = ConfigStore()
    cs.store(name='eeg', node=EEGLearnerConfig)
    main()
