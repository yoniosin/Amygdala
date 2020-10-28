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
from optuna.integration import PyTorchLightningPruningCallback


class MetricCallback(pl.Callback):
    def __init__(self):
        self.metric = None
        self.is_sanity = True

    def on_validation_epoch_end(self, trainer:pl.Trainer, pl_module):
        if self.is_sanity:
            self.is_sanity = False
        else:
            self.metric = (trainer.callback_metrics['val_loss'])


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
    cfg.net.lr = 5e-1
    metric_callback = MetricCallback()
    data.phase = 3
    trainer = create_trainer(cfg, ['indices'], callbacks=[metric_callback])

    trainer.fit(model, datamodule=data)
    return metric_callback.metric


def create_trainer(cfg, tags=None, trial=None, callbacks=None):
    if trial:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(f'trial#{trial.number}')
        new_callbacks = [PyTorchLightningPruningCallback(trial, 'val_loss')]
        if callbacks:
            new_callbacks.extend(callbacks)

        trainer = pl.Trainer(
            logger=False,
            callbacks=new_callbacks,
            checkpoint_callback=checkpoint_callback,
            max_epochs=400,
            progress_bar_refresh_rate=0,
            weights_summary=None
        )
    else:
        trainer = pl.Trainer(
            logger=NeptuneLogger(
                project_name='yoniosin/amygdala',
                tags=tags,
                params=flatten(cfg, reducer='path')
            ),
            max_epochs=cfg.learner.max_epochs,
            # callbacks=[pl.callbacks.EarlyStopping('val_loss', patience=200)]
            # fast_dev_run=True
        )

    return trainer


def full_train(cfg, data):
    first_phase_model = train_first_phase(cfg, data)

    second_phase_model: Net.EEGNetwork = train_second_phase(cfg, data, first_phase_model)
    # second_phase_model = Net.EEGNetwork(**cfg.net)
    # ckpt = torch.load(
    #     r'C:\\Users\\yonio\\PycharmProjects\\Amygdala_new\\outputs\\2020-10-20\\23-14-28\\Untitled\\AM-281\\checkpoints\\epoch=499.ckpt'
    # )
    # second_phase_model.load_state_dict(ckpt['state_dict'])

    third_phase_model = Net.IndicesNetwrork(
        second_phase_model.regulate_enc.embedding_lut, cfg.data.criteria_len, bins_num=3
    )
    last_metric = train_third_phase(cfg, data, third_phase_model)

    return last_metric


@hydra.main(config_name='eeg')
def main(cfg: EEGLearnerConfig):
    data: EEGDataModule = EEGDataModule(cfg)
    if cfg.full_train:
        full_train(cfg, data)
    elif cfg.validation:
        ckpt = torch.load(
            r'C:\Users\yonio\PycharmProjects\Amygdala_new\outputs\2020-10-25\23-43-07\Untitled\AM-374\checkpoints\epoch=265.ckpt'
        )

        lut = Net.EEGNetwork(**cfg.net).regulate_enc.embedding_lut
        model = Net.IndicesNetwrork(lut, cfg.data.criteria_len, 3)
        model.load_state_dict(ckpt['state_dict'])

        data.phase = 3

        trainer = create_trainer(cfg, ['indices'])
        trainer.test(model, datamodule=data)
    else:
        ckpt = torch.load(
            r'C:\Users\yonio\PycharmProjects\Amygdala_new\outputs\2020-10-25\01-03-56\Untitled\AM-328\checkpoints\epoch=499.ckpt'
        )
        eeg_model = Net.EEGNetwork(**cfg.net)
        eeg_model.load_state_dict(ckpt['state_dict'])
        model = Net.IndicesNetwrork(
            eeg_model.regulate_enc.embedding_lut, cfg.data.criteria_len, bins_num=3
        )

        train_third_phase(cfg, data, model)


if __name__ == '__main__':
    cs = ConfigStore()
    cs.store(name='eeg', node=EEGLearnerConfig)
    main()
