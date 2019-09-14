import torch.nn as nn
import torch
import pickle
import re
from pathlib import Path
import progressbar
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.autograd import Variable

def get_score_dict():
    def score(subject):
        first_pw = subject.paired_windows[0]
        return [first_pw.watch_window.mean, first_pw.regulate_window.mean]

    subjects_dir_path = Path('/home/yonio/Projects/conv_gru/3d_data/3D')
    score_dict = {}

    for subject_path in subjects_dir_path.iterdir():
        subject = pickle.load(open(str(subject_path), 'rb'))
        
        score_dict[subject_num] = score(subject)
    
    return score_dict



class Regressor(nn.Module):
    def __init__(self, emebed_size, out_size, train_dl, test_dl):
        return super().__init__()
        self.fc = nn.Linear(emebed_size, out_size)
        self.score_dict = get_score_dict()
    
    def train(self, n_epochs):
        bar = progressbar.ProgressBar()
        writer = SummaryWriter(self.run_name)
        for i in bar(range(n_epochs)):
            train_loss = self.run_model(train=True)
            writer.add_scalar('train', np.mean([x[0] for x in train_loss]), i)
            test_loss = self.test()
            writer.add_scalar('test', np.mean([x[0] for x in test_loss]), i)
    
    def test(self):
        with torch.no_grad():
            return self.run_model(train=False)

    def run_model(self, train: bool):
        dl = self.train_dl if train else self.test_dl
        return [self.run_batch(batch, train) for batch in dl]

    def run_batch(self, batch, train: bool):
        x, _, subject_id, subject_name = batch
        ds_shape = y.shape
        y = y.view(ds_shape[0], *ds_shape[2:-1], ds_shape[1] * ds_shape[-1])
        input_ = Variable(x, requires_grad=train)
        target = Variable(y, requires_grad=False)
        output, c = self.net(input_, subject_id, y)

        loss = self.loss_func(output, target)
        transition_loss = self.loss_func(output[..., self.transition_phases], target[..., self.transition_phases])
        if train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return float(loss), float(transition_loss)



if __name__ == '__main__':
    ds_location = Path('data')
    train_dl = torch.load(ds_location / 'train.pt')
    test_dl = torch.load(ds_location / 'test_o.pt')
