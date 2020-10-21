from util.config import EEGLearnerConfig
import Models.Networks as Net
import hydra
from hydra.core.config_store import ConfigStore
from flatten_dict import flatten
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from data.eeg.eeg_data_module import EEGDataModule
from data.subject import EEGSubjectPTSD, EEGWindow, PairedWindows  # needed for pickling
import torch


def train_first_phase(cfg, data, model=None):
    """In the first phase, all of the components are training. Only training data is introduced"""
    model = model or Net.EEGNetwork(**cfg.net)

    trainer = create_trainer(cfg, ['First'].append(('Embed' if cfg.net.embedding_size > 0 else 'vanilla')))
    trainer.fit(model, datamodule=data)
    return model


def train_second_phase(cfg, data: EEGDataModule, model: Net.EEGNetwork, state_dict=None):
    """In the second phase, only the LUT is trained, and we can therefore introduce the validation subjects"""
    if state_dict:
        model.load_state_dict(state_dict)
    data.phase = 2
    model.first_phase_requires_grad(False)

    trainer = create_trainer(cfg, ['Second'])
    trainer.fit(model, datamodule=data)

    return model


def train_third_phase(cfg, data: EEGDataModule, model):
    cfg.learner.max_epochs = 10000
    data.phase = 3
    trainer = create_trainer(cfg, ['indices'])

    trainer.fit(model, datamodule=data)


def create_trainer(cfg, tags=None):
    trainer = pl.Trainer(
        logger=NeptuneLogger(
            project_name='yoniosin/amygdala',
            tags=tags,
            params=flatten(cfg, reducer='path')
        ),
        max_epochs=cfg.learner.max_epochs,
        # fast_dev_run=True
    )

    return trainer


@hydra.main(config_name='eeg')
def main(cfg: EEGLearnerConfig):
    data: EEGDataModule = EEGDataModule(cfg)
    # first_phase_model = train_first_phase(cfg, data)

    # second_phase_model: Net.EEGNetwork = train_second_phase(cfg, data, first_phase_model)
    second_phase_model = Net.EEGNetwork(**cfg.net)
    ckpt = torch.load(
        r'C:\\Users\\yonio\\PycharmProjects\\Amygdala_new\\outputs\\2020-10-20\\23-14-28\\Untitled\\AM-281\\checkpoints\\epoch=499.ckpt'
    )
    second_phase_model.load_state_dict(ckpt['state_dict'])

    third_phase_model = Net.IndicesNetwrork(
        second_phase_model.regulate_enc.embedding_lut, cfg.data.criteria_len
    )
    train_third_phase(cfg, data, third_phase_model)


if __name__ == '__main__':
    cs = ConfigStore()
    cs.store(name='eeg', node=EEGLearnerConfig)
    main()
