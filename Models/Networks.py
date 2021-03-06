from torch import nn
import torch
from Models.EmbeddingLSTM import EmbeddingLSTM, EEGEmbedingLSTM
from itertools import chain
import pytorch_lightning as pl


class SequenceTransformNet(nn.Module):
    """
    Using a Single Transformation, this module maps a series of [Active, Passive]+ last Passive to the last Active
    [P P
          + P ---> A
     A A]
    """
    def __init__(self, input_shape, hidden_size, output_size, use_embeddings, n_subjects):
        super().__init__()
        self.single_transform = torch.load(open('single_run.pt', 'rb'))  # load previously trained ST
        phases, _, height, width, depth, phase_len = input_shape
        seq_len = phases * phase_len
        self.out_sizes = [height, width, depth, seq_len]
        spacial_size = height * width * depth

        # create Seq2Seq NN
        # 2 factor is because we concat recent history [Xt|Yt-1]
        self.rnn = EmbeddingLSTM(2 * spacial_size, [hidden_size], n_subjects, use_embeddings)
        self.fc1 = nn.Linear(seq_len * hidden_size, seq_len * spacial_size)
        self.fc2 = nn.Linear(seq_len, output_size[-1])

    @property
    def out_features(self): return self.fc1.out_features

    def forward(self, x, subject_id, y):
        split = torch.split(x, 1, dim=1)
        xT = torch.cat([self.single_transform(x_i) for x_i in split], dim=-1)
        rnn_out, c_out = self.rnn(xT.squeeze(1), subject_id, y)
        y = self.fc1(rnn_out.view(x.shape[0], -1)).view(x.shape[0], *self.out_sizes)
        return y.squeeze(-1), c_out


class NewSTNet(nn.Module):
    """This Module finds a mapping between passive windows to active windows, using denoising AutoEncoders"""
    def __init__(self, **kwargs):
        super().__init__()
        input_shape = kwargs['input_shape']
        sizes = [1, 2, 4, 8, 8, 4, 2, 1]
        self.layers_n = len(sizes) - 1
        input_size = input_shape[-1]
        for i in range(self.layers_n):
            in_size = sizes[i]
            out_size = sizes[i+1]
            setattr(self, f'l{i}', nn.Linear(input_size * in_size, input_size * out_size))

    def forward(self, x):
        for i in range(self.layers_n):
            linear_i = getattr(self, f'l{i}')
            activation_i = nn.Dropout() if i % 2 == 0 else nn.ReLU()
            x = activation_i((linear_i(x)))

        return x


class ClassifyingNetwork(nn.Module):
    def __init__(self, n_classes, n_filters, conv_kernel_size, embedding_size, n_windows, seq_len):
        super().__init__()
        self.n_windows = n_windows
        self.seq_len = seq_len
        channels_n = 2  # 2 in_channels are (Watch, Regulate)
        self.conv = nn.Conv3d(channels_n, n_filters, conv_kernel_size)
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
            spacial_dim = list(range(len(data.shape)))[-3:]  # last 3 dimensions are to be reduced
            return data.mean(dim=spacial_dim)
        conv_out = torch.stack([global_average_pooling(self.conv(x[..., t])) for t in range(self.seq_len)], dim=1)
        mlp_out = self.mlp(conv_out).view(x.shape[0], -1)
        return nn.Softmax(dim=1)(self.final_fc(mlp_out))


class StatisticalLinearBaseLine(nn.Module):
    """
    Receives varied size images.
    If n_outputs == 1, performs regression.
    Otherwise, classifies to n_outputs classes
    """

    def __init__(self, n_outputs, seq_len, n_windows, input_channels=2, stats_per_time=2):
        """
        input_channels - watch + regulate
        stats_per_time- mean + variance
        """
        super().__init__()
        self.n_windows = n_windows
        self.fc = nn.Linear(input_channels * stats_per_time * seq_len, n_outputs)

    def forward(self, batch):
        """
        Calculate variance and mean for every timestamp. Concatenates to [batch_size, 2] tensor,
        which is fed to a FC layer
        """
        def calc_linear_stats(dims):
            def calc_stats(s):
                mean = torch.mean(s, dim=dims)
                var = torch.var(s, dim=dims)

                return mean, var

            mean_list, var_list = zip(*[calc_stats(subject) for subject in batch])
            res = torch.cat((torch.stack(mean_list), torch.stack(var_list)), dim=1)
            return res

        all_results = calc_linear_stats((1, 2, 3))
        y = self.fc(all_results.view(all_results.shape[0], -1))
        return y


class CNNBaseline(nn.Module):
    def __init__(self, filter_n, mlp_len, n_outputs):
        super().__init__()
        self.conv = nn.Conv3d(56, filter_n, kernel_size=3)
        final_layer = nn.Linear(filter_n, n_outputs)
        self.mlp = nn.Sequential(*chain(*[(nn.Linear(filter_n, filter_n), nn.ReLU()) for _ in range(mlp_len)]),
                                 final_layer)

    def forward(self, x):
        x = torch.cat((x[:, 0], x[:, 1]), dim=-1).transpose(-1, 1)
        z = self.conv(x)
        z = torch.mean(z, dim=(2, 3, 4))
        return self.mlp(z)


class EEGNetwork(pl.LightningModule):
    def __init__(self,
                 watch_len,
                 reg_len,
                 watch_hidden_size,
                 reg_hidden_size,
                 embedding_size,
                 lr,
                 train_lstm: bool = True,
                 weight_decay=0,
                 n_subjects=None):
        super().__init__()
        self.reg_len = reg_len
        self.watch_hidden_size = watch_hidden_size
        self.embedding_size = embedding_size
        self.train_lstm = train_lstm

        self.watch_enc = nn.Sequential(
            nn.Linear(watch_len, reg_len * watch_hidden_size).float(),
            nn.ReLU()
        )

        self.regulate_enc = EEGEmbedingLSTM(reg_hidden_size, embedding_size, n_subjects).float()
        self.regulate_enc.lstm.requires_grad_(train_lstm)

        self.fc = nn.Linear(watch_hidden_size + reg_hidden_size, 1).float()
        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def first_phase_requires_grad(self, requires_grad: bool):
        """Freeze / un-freeze first phase components"""
        self.train_lstm = requires_grad
        self.watch_enc.requires_grad_(requires_grad)
        self.regulate_enc.lstm.requires_grad_(requires_grad)

    def forward(self, batch):
        watch = batch['watch'].squeeze()
        regulate = batch['regulate'].float().permute(1, 0, 2)
        subject_id = batch['id']['system_idx']

        reg_len, batch_size, _ = regulate.shape
        watch_rep = self.watch_enc(watch.float()).view((reg_len, batch_size, -1))

        regulate_rep, _ = self.regulate_enc(regulate.float(), subject_id)
        regulate_rep = nn.functional.relu(regulate_rep)

        joint_rep = torch.cat((watch_rep, regulate_rep), dim=-1).view(batch_size*reg_len, -1)
        out = self.fc(joint_rep).view(reg_len, batch_size, -1)
        loss = self.loss_fn(out, regulate)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        metric_name = ('train' if self.train_lstm else 'val') + '_loss'

        self.log(metric_name, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.regulate_enc.embedding_lut.train()

        self.log('val_loss', loss)


class IndicesNetwrork(pl.LightningModule):
    criteria_name = ['caps', 'stai', 'tas']

    def __init__(self, embedding_lut: nn.Module, criteria_num, bins_num: int = 1):
        super().__init__()
        self.embedding_lut = embedding_lut
        self.embedding_lut.requires_grad_(False)  # freeze LUT
        self.criteria_num = criteria_num
        self.fc = nn.Linear(embedding_lut.embedding_dim, criteria_num * bins_num).float()
        self.loss_fn = nn.MSELoss(reduction='none') if bins_num == 1 else nn.CrossEntropyLoss(reduction='none').float()

    def forward(self, subject_id, ground_truth):
        embeddings = self.embedding_lut(subject_id)
        prediction = self.fc(embeddings)
        prediction = prediction.view(subject_id.shape[0], -1, self.criteria_num).squeeze(1)
        loss = torch.mean(self.loss_fn(prediction, ground_truth.long()), dim=0)

        return loss

    def training_step(self, batch, batch_idx):
        ground_truth = torch.stack([batch[name] for name in self.criteria_name], dim=-1).float()
        loss = self(batch['system_idx'], ground_truth)
        self.log_results(loss, True)
        loss = torch.mean(loss)
        self.log('train_loss', loss)

        return loss

    def log_results(self, loss, train: bool):
        for idx, name in enumerate(self.criteria_name):
            metric_name = ('train' if train else 'val') + f'_{name}'
            self.log(metric_name, torch.mean(loss[idx]))

    def validation_step(self, batch, batch_idx):
        ground_truth = torch.stack([batch[name] for name in self.criteria_name], dim=-1).float()
        loss = self(batch['system_idx'], ground_truth)

        self.log_results(loss, False)
        self.log('val_loss', torch.mean(loss), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)
