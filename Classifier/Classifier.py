import torch
from torch import nn, optim
import numpy as np
import progressbar
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from sklearn.svm import SVR
from typing import Iterable
from functools import reduce, partial
from more_itertools import first_true
from abc import abstractmethod
from Models.SingleTransform import StatisticalLinearBaseLine
from collections import Counter


class EmbeddingPredictor:
    def __init__(self, train_dl, test_dl, embedding_layer, meta_data_iter: Iterable, n_outputs, feature, run_num):
        def get_dicts(dict_type: str):
            return reduce(lambda x, y: {**x, **y}, map(lambda t: getattr(t, dict_type), meta_data_iter))

        self.run_num = run_num
        self.predicted_feature = feature
        self._embed = embedding_layer
        self.n_outputs = n_outputs
        self.net = self.generate_NN()
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.optimizer = optim.Adam(self.net.parameters(), lr=5e-3, weight_decay=0)

        self.meta_data_iter = meta_data_iter
        for dict_type_str in ('train', 'test', 'full'):
            needed_str = f'{dict_type_str}_dict'
            setattr(self, needed_str, get_dicts(needed_str))

    def generate_NN(self): return nn.Linear(self._embed.out_features, self.n_outputs)

    def get_run_num(self): return self.run_num

    @property
    def loss_func(self): return nn.MSELoss() if self.n_outputs == 1 else nn.CrossEntropyLoss()

    def calc_loss(self, target, output, train: bool):
        loss = self.loss_func(output, target)
        if train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss

    def get_embedding(self, one_hot):
        return self._embed(one_hot)

    def train(self, n_epochs):
        bar = progressbar.ProgressBar()
        writer = SummaryWriter(f'binned_runs/{self.predicted_feature}_{self.get_run_num()}')
        for epoch in bar(range(n_epochs)):
            train_stats = self.run_model(train=True)
            test_stats = self.test()
            self.update_logger(writer, train_stats, test_stats, epoch)

        writer.close()

    def update_logger(self, writer, train_stats, test_stats, epoch):
        writer.add_scalar('train_loss', torch.mean(torch.tensor(train_stats)), epoch)
        writer.add_scalar('test_loss', torch.mean(torch.tensor(test_stats)), epoch)

    def run_model(self, train: bool):
        dl = self.train_dl if train else self.test_dl
        res = []
        for batch in dl:
            res.append(self.run_batch(batch, train))

        return res

    def run_batch(self, batch, train: bool):
        input_ = self.get_input_from_batch(batch, train)
        target = self.construct_target_tensor(batch)
        output = self.net(input_)

        return self.calc_loss(output.squeeze(), target, train)

    def get_input_from_batch(self, batch, train: bool):
        return Variable(self.get_embedding(batch['one_hot']), requires_grad=train)

    def test(self):
        with torch.no_grad():
            return self.run_model(train=False)

    def construct_target_tensor(self, batch):
        t = [self.calc_target(float(self.full_dict[s][self.predicted_feature])) for s in batch['sub_num']]
        return Variable(torch.tensor(t), requires_grad=False)

    def calc_target(self, x): return x

    @property
    def all_values(self):
        return [self.calc_target(float(self.full_dict[s][self.predicted_feature])) for s in self.full_dict.keys()]

    def dummy_prediction_error(self): return self.all_values.var()

    def svr_prediction_error(self):
        def create_svr_labels(train: bool):
            res_dict = self.train_dict if train else self.test_dict
            aux_features = {'TAS', 'STAI'}.difference({self.predicted_feature})
            x = [[float(res_dict[key][feature]) for feature in aux_features] for key in res_dict.keys()]
            y = [float(res_dict[key][self.predicted_feature]) for key in res_dict.keys()]

            return np.array(x), np.array(y)

        (train_x, train_y), (test_x, test_y) = list(map(create_svr_labels, (True, False)))
        model = SVR(gamma='auto')
        model.fit(train_x, train_y)
        y_hat = model.predict(test_x)
        return np.mean((test_y - y_hat) ** 2)


class EmbeddingClassifier(EmbeddingPredictor):
    def __init__(self, train_dl, test_dl, embedding_layer, meta_data_iter: Iterable, feature, run_num, n_outputs=5):
        super().__init__(train_dl, test_dl, embedding_layer, meta_data_iter, n_outputs, feature, run_num)

        def bin_creator():
            def generate_label(actual_value):
                """
                Given bins' boundary, returns the smallest boundary which is smaller than the value.
                If none exists, the last bin is chosen
                 """
                return first_true(range(len(bins)), pred=lambda i: actual_value < bins[i], default=len(bins) - 1)

            if n_outputs % 2 == 0: raise ValueError('Only odd bins number')
            all_values = np.array([float(self.full_dict[s][feature]) for s in self.full_dict.keys()])
            mean = np.mean(all_values)
            std = np.std(all_values)
            bins_per_side = int(n_outputs / 2)
            bins = list(map(lambda a: mean + (0.5 + a) * std / 10, range(-bins_per_side, bins_per_side)))

            return generate_label
        self.label_generator = bin_creator() if n_outputs > 1 else lambda x: x

    def calc_loss(self, output, target, train: bool):
        prediction = torch.argmax(output, dim=1)
        accuracy = torch.mean((prediction == target).float())
        return float(super().calc_loss(target, output, train)), accuracy

    def calc_target(self, x):
        return self.label_generator(x)

    def update_logger(self, writer, train_stats, test_stats, epoch):
        """Separate loss and accuracy"""
        train_loss, train_accuracy = list(zip(*train_stats))
        test_loss, test_accuracy = list(zip(*test_stats))

        super().update_logger(writer, train_loss, test_loss, epoch)
        writer.add_scalar('train_accuracy', np.mean(train_accuracy), epoch)
        writer.add_scalar('test_accuracy', np.mean(test_accuracy), epoch)

    def dummy_prediction_error(self):
        counter = Counter(self.all_values)
        greatest_chance = counter.most_common(1)[0][1] / len(self.all_values)
        return greatest_chance


class EmbeddingClassifierBaseline(EmbeddingClassifier):
    def generate_NN(self):
        return StatisticalLinearBaseLine(self.n_outputs, 28, 2)

    def get_input_from_batch(self, batch, train: bool):
        data = batch['data']
        data = torch.cat([data[:, i] for i in range(data.shape[1])], dim=-1)
        return Variable(data, requires_grad=train)

    def get_run_num(self): return f'base_{super().get_run_num()}'

    def train(self, n_epochs):
        print(f'    dummy error = {self.dummy_prediction_error()}')
        print(f'    svr error = {self.svr_prediction_error()}')
        super().train(n_epochs)

