import torch
from torch import nn, optim
import numpy as np
import progressbar
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from sklearn.svm import SVR
from typing import Iterable
from functools import reduce


class BaseRegression:
    def __init__(self, train_dl, test_dl, embedding_layer, meta_data_iter: Iterable, n_outputs):
        def get_dicts(dict_type):
            return reduce(lambda x, y: {**x, **y}, map(lambda t: getattr(t, dict_type), meta_data_iter))

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

        self.meta_data_iter = meta_data_iter
        self.train_dict = get_dicts('train_dict')
        self.test_dict = get_dicts('test_dict')
        self.meta_data = get_dicts('full_dict')

    def get_score(self, name, predicted_feature):
        try:
            res = self.meta_data[name][predicted_feature]
            return float(res)
        except ValueError as e:
            raise e

    def get_embedding(self, one_hot):
        return self._embed(one_hot)

    def fit(self, n_epochs, predicted_feature):
        bar = progressbar.ProgressBar()
        writer = SummaryWriter(f'runs/{predicted_feature}1')
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
        t = [self.get_score(s, predicted_feature) for s in batch['sub_num']]
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

    def dummy_prediction_error(self, feature):
        all_grades = np.array([self.get_score(s, feature) for s in self.meta_data.keys()])
        var = all_grades.var()
        return var

    def svr_prediction_error(self, predicted_feature, aux_features):
        def create_svr_labels(train: bool):
            res_dict = self.train_dict if train else self.test_dict
            x = [float(res_dict[key][feature]) for feature in aux_features for key in res_dict.keys()]
            y = [float(res_dict[key][predicted_feature]) for key in res_dict.keys()]

            return np.reshape(np.array(x), (-1, 1)), np.reshape(np.array(y), (-1, 1))

        (train_x, train_y), (test_x, test_y) = list(map(create_svr_labels, (True, False)))
        model = SVR(gamma='auto')
        model.fit(train_x, train_y)
        y_hat = model.predict(test_x)
        return np.mean((test_y - y_hat) ** 2)

