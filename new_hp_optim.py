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
from util.config import EEGLearnerConfig
import main_eeg


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

    def __init__(self, cfg: EEGLearnerConfig, trial: Trial):
        watch_hidden_size = trial.suggest_int('watch_hidden_size', 1, 15)
        reg_hidden_size = trial.suggest_int('reg_hidden_size', 1, 15)
        embedding_size = trial.suggest_int('embedding_size', 0, 15)
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)

        super().__init__(
            cfg.net.watch_len,
            cfg.net.reg_len,
            watch_hidden_size,
            reg_hidden_size,
            embedding_size,
            lr,
            n_subjects=cfg.net.n_subjects
        )

class EEGData(pl.LightningDataModule):
    def __init__(self, ds_location, batch_size):
        super().__init__()
        self.ds_location = ds_location
        self.batch_size = batch_size

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return torch.load(str(Path(self.ds_location) / 'train.pt'))

    def val_dataloader(self, *args, **kwargs):
        return torch.load(str(Path(self.ds_location) / 'test.pt'))


@hydra.main(config_name='eeg')
def main(cfg: EEGLearnerConfig):
    def objective(trial: Trial):
        return main_eeg.main(cfg)

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
