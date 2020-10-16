from util.config import EEGLearnerConfig
import Models.Networks as Net
import hydra
from hydra.core.config_store import ConfigStore
from flatten_dict import flatten
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from data.eeg.eeg_data_module import FirstPhaseEEGData, SecondPhaseEEGData
from util.Subject import *


def train_first_phase(cfg, model=None):
    """In the first phase, all of the components are training. Only training data is introduced"""
    data = FirstPhaseEEGData(cfg)
    model = model or Net.EEGNetwork(**cfg.net)

    trainer = create_trainer(cfg, ['First'].append(('Embed' if cfg.net.embedding_size > 0 else 'vanilla')))
    trainer.fit(model, datamodule=data)
    return model


def train_second_phase(cfg, model: Net.EEGNetwork, state_dict=None):
    """In the second phase, only the LUT is trained, and we can therefore introduce the validation subjects"""
    data = SecondPhaseEEGData(cfg)
    if state_dict:
        model.load_state_dict(state_dict)
    model.first_phase_requires_grad(False)

    trainer = create_trainer(cfg, ['Second'])
    trainer.fit(model, datamodule=data)


def create_trainer(cfg, tags):
    trainer = pl.Trainer(
        logger=NeptuneLogger(
            project_name='yoniosin/amygdala',
            tags=tags,
            params=flatten(cfg, reducer='path')
        ),
        max_epochs=cfg.learner.max_epochs
    )

    return trainer


@hydra.main(config_name='eeg')
def main(cfg: EEGLearnerConfig):
    model = train_first_phase(cfg)
    train_second_phase(cfg, model)


if __name__ == '__main__':
    cs = ConfigStore()
    cs.store(name='eeg', node=EEGLearnerConfig)
    main()
