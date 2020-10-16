import pytorch_lightning as pl
import Models.Networks as Net
from optuna.trial import Trial
import optuna
import hydra
from hydra.core.config_store import ConfigStore
from util.config import EEGLearnerConfig
import json
import torch
from torch.utils.data import DataLoader
from pathlib import Path
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


class EEGNetHPO(Net.EEGNetwork):
    watch_len = None
    reg_len = None
    n_subjects = None

    def __init__(self, trial: Trial):
        watch_hidden_size = trial.suggest_int('watch_hidden_size', 1, 15)
        reg_hidden_size = trial.suggest_int('reg_hidden_size', 1, 15)
        embedding_size = trial.suggest_int('embedding_size', 0, 15)
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)

        super().__init__(
            self.watch_len,
            self.reg_len,
            watch_hidden_size,
            reg_hidden_size,
            embedding_size,
            lr,
            n_subjects=self.n_subjects
        )


def objective(trial: Trial):
    model = EEGNetHPO(trial)
    data = EEGData()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(f'trial#{trial.number}')
    metrics_callback = MetricCallback()
    trainer = pl.Trainer(
        logger=False,
        callbacks=[metrics_callback, PyTorchLightningPruningCallback(trial, 'val_loss')],
        checkpoint_callback=checkpoint_callback,
        max_epochs=400,
        progress_bar_refresh_rate=0,
        weights_summary=None
    )

    trainer.fit(model, datamodule=data)
    return metrics_callback.metric


class EEGData(pl.LightningDataModule):
    batch_size = None
    ds_location = None

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return torch.load(str(Path(self.ds_location) / 'train.pt'))

    def val_dataloader(self, *args, **kwargs):
        return torch.load(str(Path(self.ds_location) / 'test.pt'))


@hydra.main(config_name='eeg')
def main(cfg: EEGLearnerConfig):
    EEGNetHPO.reg_len = cfg.net.reg_len
    EEGNetHPO.watch_len = cfg.net.watch_len
    EEGNetHPO.n_subjects = cfg.net.n_subjects

    EEGData.batch_size = cfg.learner.batch_size
    EEGData.ds_location = str(Path(cfg.learner.main_dir) / 'data/eeg')

    study = optuna.create_study(
        direction='minimize', pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=100)
    best_trial = study.best_trial
    print(f'best result is {best_trial.value}')
    json.dump(best_trial.params, open('best_trial.json'), 'wb')


if __name__ == '__main__':
    cs = ConfigStore()
    cs.store(name='eeg', node=EEGLearnerConfig)
    main()
