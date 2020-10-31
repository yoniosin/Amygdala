import pytorch_lightning as pl
from torch import nn
from data.eeg.eeg_data_module import EEGDataModule
import hydra
from hydra.core.config_store import ConfigStore
from util.config import EEGLearnerConfig
from torch.optim import Adam
import torch
from torch.nn.functional import softmax


class DomainAdaptation(pl.LightningModule):
    def __init__(self, source_lut: nn.Embedding, target_lut: nn.Embedding):
        super().__init__()
        # make sure LUTs are not trained
        for lut in (source_lut, target_lut):
            lut.requires_grad_(False)
        self.source_lut = source_lut
        self.target_lut = target_lut

        self.adaptor = nn.Linear(source_lut.embedding_dim, target_lut.embedding_dim)
        self.loss = nn.MSELoss()
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, subject_id, train: bool):
        emb = self.source_lut(subject_id)
        prediction = self.adaptor(emb)

        ground_truth = self.target_lut(subject_id)
        loss = self.loss(prediction, ground_truth)
        name = f'{"train" if train else "val"}_loss'

        self.log(name, loss)

        return loss if train else None

    def training_step(self, batch, _):
        return self(self.extract_id(batch), train=True)

    def validation_step(self, batch, _):
        return self(self.extract_id(batch), train=False)

    @staticmethod
    def extract_id(batch): return batch['id']['system_idx']

    def test_step(self, batch, _):
        sub1_source, sub2_source = self.source_lut(self.extract_id(batch)).split(1)
        sub1_truth, sub2_truth = self.target_lut(self.extract_id(batch)).split(1)
        sub1_target = self.adaptor(sub1_source)

        same_person = self.loss(sub1_target, sub1_truth)
        different_person = self.loss(sub1_target, sub2_truth)
        logits = softmax(torch.tensor([same_person, different_person]))
        acc = self.accuracy(logits, torch.tensor([1, 0]))

        self.log('same person', same_person)
        self.log('different person', different_person)
        self.log('accuracy', acc)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)


@hydra.main(config_name='eeg')
def main(cfg: EEGLearnerConfig):
    cfg.correlation = True
    cfg.data.load = True

    data = EEGDataModule(cfg, use_criteria=True)
    model = DomainAdaptation(nn.Embedding(164, 10), nn.Embedding(164, 5))
    trainer = pl.Trainer()

    trainer.fit(model, datamodule=data)

    trainer.test(model, datamodule=data)


if __name__ == '__main__':
    cs = ConfigStore()
    cs.store(name='eeg', node=EEGLearnerConfig)
    main()
