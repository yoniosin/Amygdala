import torch
from torch import nn, optim
import numpy as np
import progressbar
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable


class BaseRegression:
    def __init__(self, train_dl, test_dl, embedding_layer, meta_data, n_outputs):
        self._embed = embedding_layer
        self.n_outputs = n_outputs
        layers = [nn.Linear(embedding_layer.out_features, n_outputs)]
        if n_outputs > 1:
            layers.append(nn.Softmax())
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_func = nn.MSELoss()
        self.fc = nn.Sequential(*layers)
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.optimizer = optim.Adam(self.fc.parameters(), lr=1e-1, weight_decay=0)

        self.meta_data = meta_data

    def get_score(self, name, predicted_feature):
        return self.meta_data[name][predicted_feature]

    def get_embedding(self, one_hot):
        return self._embed(one_hot)

    def fit(self, n_epochs, predicted_feature):
        bar = progressbar.ProgressBar()
        writer = SummaryWriter(f'runs/{predicted_feature}7')
        for i in bar(range(n_epochs)):
            train_loss = self.run_model(train=True, predicted_feature=predicted_feature)
            writer.add_scalar('train', np.mean([x[0] for x in train_loss]), i)
            # writer.add_scalar('train_acc', np.mean([x[1] for x in train_loss]), i)

            test_loss = self.test(predicted_feature)
            writer.add_scalar('test', np.mean([x[0] for x in test_loss]), i)
            # writer.add_scalar('test_acc', np.mean([x[1] for x in test_loss]), i)

    def run_model(self, train: bool, predicted_feature):
        dl = self.train_dl if train else self.test_dl
        res = []
        for batch in dl:
            res.append(self.run_batch(batch, train, predicted_feature))

        return res

    def run_batch(self, batch, train: bool, predicted_feature):
        # subject_num, x, subject_score, subject_one_hot = batch
        input_ = Variable(self.get_embedding(batch['one_hot']), requires_grad=train)
        t = [self.get_score(int(s), predicted_feature) for s in batch['sub_num']]
        target = Variable(torch.tensor(t, requires_grad=False).float())
        output = self.fc(input_)

        loss = self.loss_func(output.squeeze(), target)
        if train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        prediction = torch.argmax(output, dim=1)
        accuracy = 0 if self.n_outputs == 1 else torch.sum(prediction == target).float() / len(prediction)
        return float(loss), accuracy

    def test(self, predicted_feature):
        with torch.no_grad():
            return self.run_model(train=False, predicted_feature=predicted_feature)
