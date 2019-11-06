from util.config import LearnerMetaData
from torch import nn, optim
import progressbar
import numpy as np
import torch
from Models.EmbeddingLSTM import EmbeddingLSTM
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from abc import abstractmethod


class STNet(nn.Module):
    """This Module finds a mapping between passive windows to active windows, using a FCN"""
    def __init__(self, **kwargs):
        super().__init__()
        input_shape = kwargs['input_shape']
        sizes = [1, 2, 4, 8, 8, 4, 2, 1]
        self.layers_n = len(sizes) - 1
        for i in range(self.layers_n):
            input_size = input_shape[-1]
            in_size = sizes[i]
            out_size = sizes[i+1]
            setattr(self, f'l{i}', nn.Linear(input_size * in_size, input_size * out_size))

    def forward(self, x):
        for i in range(self.layers_n):
            linear_i = getattr(self, f'l{i}')
            activation_i = nn.Dropout() if i % 2 == 0 else nn.ReLU()
            x = activation_i((linear_i(x)))

        return x


class SequenceTransformNet(nn.Module):
    """
    Using a Single Transformation, this module maps a series of [Active, Passive]+ last Passive to the last Active
    [P P
          + P ---> A
     A A]
    """
    def __init__(self, input_shape, hidden_size, output_size, use_embeddings):
        super().__init__()
        self.single_transform = torch.load(open('single_run.pt', 'rb'))  # load preciously trained ST
        phases, _, height, width, depth, phase_len = input_shape
        seq_len = phases * phase_len
        self.out_sizes = [height, width, depth, seq_len]
        spacial_size = height * width * depth

        # create Seq2Seq NN
        # 2 factor is because we concat recent history [Xt|Yt-1]
        self.rnn = EmbeddingLSTM(2 * spacial_size, [hidden_size], use_embeddings)
        self.fc1 = nn.Linear(seq_len * hidden_size, seq_len * spacial_size)
        self.fc2 = nn.Linear(seq_len, output_size[-1])

    def forward(self, x, subject_id, y):
        split = torch.split(x, 1, dim=1)
        xT = torch.cat([self.single_transform(x_i) for x_i in split], dim=-1)
        rnn_out, c_out = self.rnn(xT.squeeze(1), subject_id, y)
        y = self.fc1(rnn_out.view(x.shape[0], -1)).view(x.shape[0], *self.out_sizes)
        # y = self.fc2(nn.ReLU()(y))
        return y.squeeze(-1), c_out


class ClassifyingNetwork(nn.Module):
    def __init__(self, n_classes, n_filter, conv_kernel_size, embedding_size, n_windows, seq_len):
        super().__init__()
        self.n_windows = n_windows
        self.seq_len = seq_len
        self.conv = nn.Conv3d(2, n_filter, conv_kernel_size)
        self.mlp = nn.Sequential(nn.Linear(n_filter, embedding_size),
                                 nn.Sigmoid(),
                                 nn.Linear(embedding_size, embedding_size))
        self.final_fc = nn.Linear(42 * embedding_size, n_classes)

    def forward(self, x):
        all_results = []
        for subject in x:
            flattened_time = torch.cat([subject[i] for i in range(self.n_windows)], dim=-1).unsqueeze(0)
            all_representations = []
            for t in range(self.seq_len):
                x1 = self.conv(flattened_time[..., t])
                all_representations.append(x1.mean(dim=(2, 3, 4)))

            conv_out = nn.Dropout(p=0.0)(torch.stack(all_representations, dim=1))
            all_representations = self.mlp(conv_out)
            res = nn.Softmax(dim=1)(self.final_fc(all_representations.view(1, -1)))
            all_results.append(res)
        all_results = torch.stack(all_results).squeeze(1)
        return all_results


class BaseModel:
    def __init__(self,  input_shape, hidden_size, md: LearnerMetaData, train_dl, test_dl, name='sqeuence'):
        self.name = name
        self.batch_size = md.batch_size
        self.n_windows = md.train_windows + 1
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.run_name = md.run_name
        self.net = self.build_NN(input_shape, hidden_size, md.use_embeddings)
        self.optimizer = optim.Adam(self.net.parameters(), lr=2e-3, weight_decay=0)

    @abstractmethod
    def build_NN(self, input_shape, hidden_size, use_embeddings): pass

    def train(self, n_epochs):
        bar = progressbar.ProgressBar()
        writer = SummaryWriter(self.run_name)
        for i in bar(range(n_epochs)):
            train_loss, train_accuracy = list(zip(*self.run_model(train=True)))
            writer.add_scalar('train_loss', np.mean(train_loss), i)
            writer.add_scalar('train_accuracy', np.mean(train_accuracy), i)
            test_loss, test_accuracy = list(zip(*self.test()))
            writer.add_scalar('test_loss', np.mean(test_loss), i)
            writer.add_scalar('test_accuracy', np.mean(test_accuracy), i)

        writer.close()
        torch.save(self.net, f'{self.name}_last_run.pt')

    def test(self):
        with torch.no_grad():
            return self.run_model(train=False)

    def run_model(self, train: bool):
        self.net.train(train)
        dl = self.train_dl if train else self.test_dl
        return [self.run_batch(batch, train) for batch in dl]

    def run_batch(self, batch, train: bool):
        output, target = self.calc_signals(batch, train)
        loss = self.optimizer_step(output, target, train)
        accuracy = list(map(lambda o, t: torch.argmax(o) == torch.argmax(t), output, target))

        return float(loss), np.mean(accuracy)

    @abstractmethod
    def calc_signals(self, batch, train): pass

    def optimizer_step(self, output, target, train):
        loss = self.loss_func(output, target)
        if train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss


class ReconstructingModel(BaseModel):
    def __init__(self, input_shape, hidden_size, md: LearnerMetaData, train_dl, test_dl):
        self.loss_func = nn.MSELoss()
        super().__init__(input_shape, hidden_size, md, train_dl, test_dl)

    def calc_signals(self, batch, train):
        x = batch['passive']
        y = batch['active']
        y = torch.cat([y[:, i] for i in range(self.n_windows)], dim=-1)  # concat active windows to long sequence
        input_ = Variable(x, requires_grad=train)
        target = Variable(y, requires_grad=False)
        output, c = self.net(input_, batch['one_hot'], y)

        return output, target

    def build_NN(self, input_shape, hidden_size, use_embeddings):
        return SequenceTransformNet(input_shape, hidden_size,input_shape, use_embeddings)


class ClassifyingModel(BaseModel):
    def __init__(self, input_shape, hidden_size, md: LearnerMetaData, train_dl, test_dl):
        self.loss_func = nn.BCELoss()
        super().__init__(input_shape, hidden_size, md, train_dl, test_dl)

    def build_NN(self, input_shape, hidden_size, use_embeddings):
        return ClassifyingNetwork(2, 10, 3, 6, n_windows=3, seq_len=42)

    def calc_signals(self, batch, train):
        def create_target(label):
            if label == 'healthy': return torch.tensor((1, 0))
            else: return torch.tensor((0, 1))

        target = torch.stack([create_target(label) for label in batch['type']]).float()
        return self.net(batch['data']), target
