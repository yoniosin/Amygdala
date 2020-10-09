from util.config import fMRILearnerConfig
from torch import nn, optim
import progressbar
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from abc import abstractmethod, ABC
from Models import Networks as Net
import numpy as np
from sklearn.svm import SVR
from typing import Iterable
from functools import reduce
from more_itertools import first_true
from Models.Networks import StatisticalLinearBaseLine
from collections import Counter


class BaseModel(ABC):
    """ Wrapper which defines high-level training procedure and data saving """
    def __init__(self, train_dl, test_dl, run_name, run_logger_path, **net_params):
        self.run_logger_path = run_logger_path
        self.run_name = run_name
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.net = self.build_NN(**net_params)
        self.optimizer = optim.Adam(self.net.parameters(), lr=5e-4, weight_decay=0)

    @abstractmethod
    def build_NN(self, **kwargs): pass

    def update_logger(self, writer, train_stats, test_stats, epoch):
        writer.add_scalar('train_loss', np.mean(train_stats), epoch)
        print(f'epoch# {epoch}, train error = {np.mean(train_stats)}')
        writer.add_scalar('test_loss', np.mean(test_stats), epoch)
        print(f'epoch# {epoch}, test error = {np.mean(test_stats)}')

    def train(self, n_epochs):
        bar = progressbar.ProgressBar()
        writer = SummaryWriter(self.run_logger_path)
        for epoch in bar(range(n_epochs)):
            train_stats = self.run_model(train=True)
            test_stats = self.test()

            self.update_logger(writer, train_stats, test_stats, epoch)

        writer.close()
        torch.save(self.net, f'trained models/{self.run_name}.pt')

    def test(self):
        with torch.no_grad():
            return self.run_model(train=False)

    def run_model(self, train: bool):
        self.net.train(train)
        dl = self.train_dl if train else self.test_dl
        return [self.run_batch(batch, train) for batch in dl]

    def run_batch(self, batch, train: bool):
        target, output = self.calc_signals(batch, train)
        return self.calc_loss(output, target, train)

    @abstractmethod
    def calc_signals(self, batch, train): pass

    @abstractmethod
    def loss_func(self): pass

    def calc_loss(self, output, target, train):
        loss = self.loss_func()(output, target)
        if train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return float(loss)


class FmriModel(BaseModel, ABC):
    """Abstract method which defines high level function for seq2seq predictors"""
    def __init__(self, md: fMRILearnerConfig, train_dl, test_dl, run_name, **net_params):
        self.n_windows = md.train_windows + 1
        super().__init__(train_dl, test_dl, run_name, run_logger_path=md.logger_path,
                         use_embeddings=md.use_embeddings, **net_params)

    def extract_part_from_data(self, data, part):
        """concat active windows to long sequence. Passive is 0, Active is 1"""
        active = data[:, :, part]
        return torch.cat([active[:, i] for i in range(self.n_windows)], dim=-1)

    def extract_passive(self, data): return self.extract_part_from_data(data, 0)


class ReconstructiveModel(FmriModel):
    def __init__(self, md: fMRILearnerConfig, train_dl, test_dl, **net_params):
        super().__init__(md, train_dl, test_dl, **net_params, run_name=f'sequence_{md.run_num}')

    def loss_func(self): return nn.MSELoss()

    def calc_signals(self, batch, train):
        x = self.extract_passive(batch['data'])
        x = Variable(x, requires_grad=train)
        y = self.extract_active(batch['data'])
        target = Variable(y, requires_grad=False)
        output = self.net(x, one_hot=batch['one_hot'], y=y)

        return output, target

    def extract_active(self, data): return self.extract_part_from_data(data, 1)

    def build_NN(self, **net_params):
        return Net.SequenceTransformNet(output_size=net_params['input_shape'], **net_params)

    def extract_passive(self, data):return data[:, :, 0]


class STModel(ReconstructiveModel):
    def __init__(self, md: fMRILearnerConfig, train_dl, test_dl, **net_params):
        super().__init__(md, train_dl, test_dl, **net_params)

    def build_NN(self, **kwargs): return Net.NewSTNet(**kwargs)

    def extract_part_from_data(self, data, part): return data[:, :, part]

    def calc_out(self, batch, **kwargs): return self.net(batch)


class ClassifyingModel(FmriModel):
    def __init__(self, md: fMRILearnerConfig, train_dl, test_dl, baseline):
        self.baseline = baseline
        super().__init__(md, train_dl, test_dl)

    def loss_func(self): return nn.BCELoss()

    def build_NN(self, **kwargs):
        return Net.StatisticalLinearBaseLine(2, 2, 28, 2) if self.baseline else Net.ClassifyingNetwork(2, 10, 3, 6, 3, 28)

    def calc_signals(self, batch, train):
        def create_target(label):
            if label == 'PTSD': return torch.tensor((1, 0))
            else: return torch.tensor((0, 1))

        data = Variable(torch.cat([batch['data'][:, i] for i in range(self.n_windows)], dim=-1), requires_grad=train)
        target = Variable(torch.stack([create_target(label) for label in batch['type']]).float(), requires_grad=False)
        return self.net(data), target

    def calc_loss(self, output, target, train):
        """In addition to the loss value, calculate accuracy as well"""
        accuracy = torch.stack(list(map(lambda o, t: (torch.argmax(o) == torch.argmax(t)), output, target))).float()
        return super().calc_loss(output, target, train), torch.mean(accuracy)

    def update_logger(self, writer, train_stats, test_stats, epoch):
        """Separate loss and accuracy"""
        train_loss, train_accuracy = list(zip(*train_stats))
        test_loss, test_accuracy = list(zip(*test_stats))

        super().update_logger(writer, train_loss, test_loss, epoch)
        writer.add_scalar('train_accuracy', np.mean(train_accuracy), epoch)
        writer.add_scalar('test_accuracy', np.mean(test_accuracy), epoch)


class EmbeddingPredictor(BaseModel):
    def __init__(self, train_dl, test_dl, embedding_layer, meta_data_iter: Iterable, n_outputs, feature, run_name):

        self.predicted_feature = feature
        self._embed = embedding_layer
        self.n_outputs = n_outputs
        self.meta_data_iter = meta_data_iter

        def get_dicts(dict_type: str):
            return reduce(lambda x, y: {**x, **y}, map(lambda t: getattr(t, dict_type), meta_data_iter))
        for dict_type_str in ('train', 'test', 'full'):
            needed_str = f'{dict_type_str}_dict'
            setattr(self, needed_str, get_dicts(needed_str))
        super().__init__(train_dl, test_dl, run_name=run_name, run_logger_path=f'binned/{run_name}')

    def loss_func(self):
        return nn.MSELoss() if self.n_outputs == 1 else nn.CrossEntropyLoss()

    def build_NN(self, **kwargs): return nn.Linear(self._embed.out_features, self.n_outputs)

    def get_embedding(self, one_hot):
        return self._embed(one_hot)

    def calc_signals(self, batch, train: bool):
        input_ = self.get_input_from_batch(batch, train)
        target = self.construct_target_tensor(batch)
        output = self.net(input_)

        return target, output

    def get_input_from_batch(self, batch, train: bool):
        return Variable(self.get_embedding(batch['one_hot']), requires_grad=train)

    def construct_target_tensor(self, batch):
        t = [self.calc_target(float(self.full_dict[s][self.predicted_feature])) for s in batch['sub_num']]
        return Variable(torch.tensor(t), requires_grad=False)

    def calc_target(self, x): return x

    @property
    def all_values(self):
        return [self.calc_target(float(self.full_dict[s][self.predicted_feature])) for s in self.full_dict.keys()]

    def dummy_prediction_error(self): return np.array(self.all_values).var()

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
    def __init__(self, train_dl, test_dl, embedding_layer, meta_data_iter: Iterable, feature, run_num, n_outputs):

        super().__init__(train_dl, test_dl, embedding_layer, meta_data_iter, n_outputs, feature,
                         run_name=f'{self.run_type}{feature}#{run_num}')

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

    @property
    def run_type(self): return ''

    def calc_loss(self, output, target, train: bool):
        prediction = torch.argmax(output, dim=1)
        accuracy = torch.mean((prediction == target).float())
        return float(super().calc_loss(output, target, train)), accuracy

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

    def get_run_num(self): return self.run_logger_path


class EmbeddingClassifierBaseline(EmbeddingClassifier):
    def __init__(self, net_type, train_dl, test_dl, embedding_layer, meta_data_iter: Iterable, feature, run_num,
                 n_outputs):
        self.net_type = net_type
        super().__init__(train_dl, test_dl, embedding_layer, meta_data_iter, feature, run_num, n_outputs)

    def build_NN(self):
        if self.net_type == 'stat':
            return StatisticalLinearBaseLine(self.n_outputs, 28, 2)
        else:
            return Net.CNNBaseline(10, 3, self.n_outputs)

    def get_input_from_batch(self, batch, train: bool):
        data = batch['data']
        data = torch.cat([data[:, i] for i in range(data.shape[1])], dim=-1)
        return Variable(data, requires_grad=train)

    def get_run_num(self): return f'base_{super().get_run_num()}'

    def train(self, n_epochs):
        print(f'    dummy error = {self.dummy_prediction_error()}')
        print(f'    svr error = {self.svr_prediction_error()}')
        super().train(n_epochs)

    @property
    def run_type(self): return 'baseline_'


class EEGModel(BaseModel):
    def __init__(self, train_dl, test_dl, run_num, logger_path, **net_params):
        super().__init__(train_dl, test_dl, run_name=f'eeg_{run_num}',
                         run_logger_path=logger_path, **net_params)

    def calc_signals(self, batch, train):
        watch = Variable(batch['watch'].squeeze(), requires_grad=train)
        regulate = Variable(batch['regulate'].float(), requires_grad=False)
        output = self.net(watch, regulate)

        return regulate, output

    def loss_func(self):
        return nn.MSELoss()

    def build_NN(self, **kwargs):
        return Net.EEGNetwork(**kwargs)

