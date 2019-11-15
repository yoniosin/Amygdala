import csv
import torch
import re
import numpy as np
from sklearn.svm import SVR
from Classifier.Classifier import BaseRegression
from pathlib import Path
import json
from util.AmygDataSet import AmygDataSet, ScoresAmygDataset
from torch.utils.data import random_split, DataLoader
from util.config import LearnerMetaData


class SubjectsMetaData:
    """
    Receives path to csv file containing meta-data for all subjects, and dumps json file ready to use.
    Dumps a list of subjects with full meta-data
    """
    def __init__(self, md_path: str, split_path: str):
        def create_subject():
            """Receives meta-data line of a subject, and returns needed values"""
            past = line['PreviousExp']
            coded_past = 2 if past == '6 EFP' else 1 if past == '1 EFP' else 0
            sex = 0 if line['Sex'] == 'M' else 1
            age = int(re.search(r'(\d{3})', line['Age']).group(1))

            return coded_past, sex, age, float(line['TAS1']), float(line['STAI_S1'])

        self.train_dict, self.test_dict = {}, {}
        self.train_data, self.test_data = None, None
        split = json.load(open(split_path, 'r'))
        self.train_list = split['train']
        self.test_list = split['test']

        with open(md_path) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for line in csv_reader:
                regex = re.search(r'(sub-(\d{3})_ses-TP(\d).*mri(\D*)_bold)', line['filename'])
                sub_name, sub_num, session_num, session_type = regex.groups()

                res_dict = self.test_dict if int(sub_num) in self.test_list else self.train_dict
                if session_type == 'practice' or session_num == 1:
                    continue
                try:
                    caps, tas, stai = (int(line['CAPS-5']), int(line['TAS']), int(line['STAI']))
                    res_dict[int(sub_num)] = {'CAPS': caps, 'TAS': tas, 'STAI': stai}
                except ValueError:
                    continue
            json.dump({**self.train_dict, **self.test_dict}, open('valid.json', 'w'))

    def create_labels(self, label_feature):
        def create_labels_for_set(train: bool):
            """Separate certain feature from the data, and set it as the label"""
            target_dict = self.train_dict if train else self.test_dict
            x, y = [], []
            keys = ['TAS', 'STAI']
            for subject in target_dict.keys():
                x.append([target_dict[subject][key] for key in keys])
                y.append(target_dict[subject][label_feature])

            res = np.array(x), np.array(y)
            if train:
                self.train_data = res
            else:
                self.test_data = res
            return res

        return list(map(create_labels_for_set, [True, False]))


def load_data_set():
    if load and Path('test_meta_dl.pt').exists() and Path('train_meta_dl.pt').exists():
        return torch.load('train_meta_dl.pt'), torch.load('test_meta_dl.pt')

    data_location = Path('../data/PTSD')
    md = LearnerMetaData(batch_size=10,
                         train_ratio=0.7,
                         run_num=100,
                         use_embeddings='init',
                         )

    ds = ScoresAmygDataset([data_location], md)
    train_ds, test_ds = random_split(ds, (ds.train_len, ds.test_len))
    train_list = [e['sub_num'] for e in train_ds]
    test_list = [e['sub_num'] for e in test_ds]
    json.dump({'train': train_list, 'test': test_list}, open('split.json', 'w'))

    train_dl_ = DataLoader(train_ds, batch_size=10, shuffle=True)
    torch.save(train_dl_, 'train_meta_dl.pt')
    test_dl_ = DataLoader(test_ds, batch_size=10, shuffle=True)
    torch.save(test_dl_, 'test_meta_dl.pt')

    return train_dl_, test_dl_


if __name__ == '__main__':
    smd = SubjectsMetaData('Clinical.csv', '../split.json')

    load = True
    train_dl, test_dl = load_data_set()
    net = torch.load('../sequence_last_run.pt')
    svr_loss = []
    for feature in ['CAPS']:
        # (train_x, train_y), (test_x, test_y) = smd.create_labels(feature)
        # model = SVR(gamma='auto')
        # model.fit(train_x, train_y)
        # y_hat = model.predict(test_x)
        # svr_loss.append(np.mean((test_y - y_hat) ** 2))
        # print(svr_loss)

        emb_reg = BaseRegression(train_dl, test_dl, net.rnn.initilaizer, {**smd.train_dict, **smd.test_dict}, 1)
        emb_reg.fit(500, feature)

    print(f'svr_loss:{svr_loss}')
