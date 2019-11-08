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
        return y.squeeze(-1), c_out


class ClassifyingNetwork(nn.Module):
    def __init__(self, n_classes, n_filters, conv_kernel_size, embedding_size, n_windows, seq_len):
        super().__init__()
        self.n_windows = n_windows
        self.seq_len = seq_len
        # 2 in_channels <==> (Passive, Active)
        self.conv = nn.Conv3d(2, n_filters, conv_kernel_size)
        self.mlp = nn.Sequential(nn.Linear(n_filters, embedding_size),
                                 nn.Sigmoid(),
                                 nn.Linear(embedding_size, embedding_size))
        self.final_fc = nn.Linear(seq_len * embedding_size, n_classes)

    def forward(self, x):
        """
        1. Pass each time step through a 3D convolution layer
        2. Average global pooling to generate fixed size tensor
        3. Fully Connected Network
        """
        def global_average_pooling(data):
            """[batch, filters, height, width, depth] --> [batch, filters] """
            return data.mean(dim=(2, 3, 4))
        conv_out = torch.stack([global_average_pooling(self.conv(x[..., t])) for t in range(self.seq_len)], dim=1)
        mlp_out = self.mlp(conv_out).view(x.shape[0], -1)
        return nn.Softmax(dim=1)(self.final_fc(mlp_out))


class ClassifyingBaseLine(nn.Module):
    """Receives varied size images and classifies the subject"""
    def __init__(self, n_classes, stats_per_time, seq_len, n_windows):
        super().__init__()
        self.n_windows = n_windows
        self.fc = nn.Linear(stats_per_time * seq_len, n_classes)

    def forward(self, batch):
        """
        Calculate variance and mean for every timestamp. Concatenates to [batch_size, 2] tensor,
        which is fed to a FC layer
        """
        def calc_stats(s):
            flattened_time = torch.cat([s[i] for i in range(self.n_windows)], dim=-1)
            mean = torch.mean(flattened_time.view(flattened_time.size(-1), -1), dim=-1)
            var = torch.var(flattened_time.view(flattened_time.size(-1), -1), dim=-1)

            return mean, var

        mean_list, var_list = zip(*[calc_stats(subject) for subject in batch])
        all_results = torch.cat((torch.stack(mean_list), torch.stack(var_list)), dim=1)
        y = nn.Softmax()(self.fc(all_results))
        return y


class BaseModel:
    def __init__(self, input_shape, hidden_size, md: LearnerMetaData, train_dl, test_dl, name, loss_func=None):
        self.name = name
        self.batch_size = md.batch_size
        self.n_windows = md.train_windows + 1
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.run_name = md.run_name
        self.net = self.build_NN(input_shape, hidden_size, md.use_embeddings)
        self.optimizer = optim.Adam(self.net.parameters(), lr=5e-4, weight_decay=0)
        self.loss_func = loss_func

    @abstractmethod
    def build_NN(self, input_shape, hidden_size, use_embeddings): pass

    def update_logger(self, writer, train_stats, test_stats, epoch):
        writer.add_scalar('train_loss', np.mean(train_stats), epoch)
        writer.add_scalar('test_loss', np.mean(test_stats), epoch)

    def train(self, n_epochs):
        bar = progressbar.ProgressBar()
        writer = SummaryWriter(self.run_name)
        for epoch in bar(range(n_epochs)):
            train_stats = self.run_model(train=True)
            test_stats = self.test()

            self.update_logger(writer, train_stats, test_stats, epoch)

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
        return self.calc_loss(output, target, train)

    @abstractmethod
    def calc_signals(self, batch, train): pass

    def calc_loss(self, output, target, train):
        loss = self.loss_func(output, target)
        if train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return float(loss)

    def extract_part_from_data(self, data, part):
        """concat active windows to long sequence. Passive is 0, Active is 1"""
        active = data[:, :, part]
        return torch.cat([active[:, i] for i in range(self.n_windows)], dim=-1)

    def extract_passive(self, data): return self.extract_part_from_data(data, 0)
    def extract_active(self, data): return self.extract_part_from_data(data, 1)


class ReconstructingModel(BaseModel):
    def calc_signals(self, batch, train):
        x = self.extract_passive(batch['data'])
        x = Variable(x, requires_grad=train)
        y = self.extract_active(batch['data'])
        target = Variable(y, requires_grad=False)
        output = self.calc_out(x, one_hot=batch['one_hot'], y=y)

        return output, target

    def __init__(self, input_shape, hidden_size, md: LearnerMetaData, train_dl, test_dl, name='sequence'):
        super().__init__(input_shape, hidden_size, md, train_dl, test_dl, name, loss_func=nn.MSELoss())

    def calc_out(self, input_, **kwargs):
        output, c = self.net(input_, kwargs['one_hot'], kwargs['y'])
        return output

    def build_NN(self, input_shape, hidden_size, use_embeddings):
        return SequenceTransformNet(input_shape, hidden_size,input_shape, use_embeddings)


class STModel(ReconstructingModel):
    def build_NN(self, input_shape, hidden_size, use_embeddings): return STNet(input_shape=input_shape)

    def extract_part_from_data(self, data, part): return data[:, :, part]

    def calc_out(self, batch, **kwargs): return self.net(batch)


class ClassifyingModel(BaseModel):
    def __init__(self, input_shape, hidden_size, md: LearnerMetaData, train_dl, test_dl, baseline):
        self.baseline = baseline
        super().__init__(input_shape, hidden_size, md, train_dl, test_dl, name='classifier', loss_func=nn.BCELoss())

    def build_NN(self, input_shape, hidden_size, use_embeddings):
        return ClassifyingBaseLine(2, 2, 42, 3) if self.baseline else ClassifyingNetwork(2, 10, 3, 6, 3, 42)

    def calc_signals(self, batch, train):
        def create_target(label):
            if label == 'healthy': return torch.tensor((1, 0))
            else: return torch.tensor((0, 1))

        data = torch.cat([batch['data'][:, i] for i in range(self.n_windows)], dim=-1)
        target = torch.stack([create_target(label) for label in batch['type']]).float()
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

