import csv
import torch
import re
import numpy as np
from sklearn.svm import SVR
from Classifier.Classifier import BaseRegression
from pathlib import Path
import json


class SubjectsMetaData:
    """Receives path to csv file containing meta-data for all subjects, and dumps json file ready to use"""
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
                subject_num = int(re.search(r'(\d{3})', line['SubCode']).group(1))
                res_dict = self.test_dict if subject_num in self.test_list else self.train_dict
                if line['TAS1'] == '' or line['STAI_S1'] == '':
                    continue
                res_dict[subject_num] = create_subject()
            json.dump({**self.train_dict, **self.test_dict}, open('valid.json', 'w'))

    def create_labels(self, label_feature):
        def create_labels_for_set(train: bool):
            """Separate certain feature from the data, and set it as the label"""
            target_dict = self.train_dict if train else self.test_dict
            x, y = [], []
            for key in target_dict.keys():
                x_before = list(target_dict[key][:label_feature])
                x_after = list(target_dict[key][label_feature + 1:])
                x.append(x_before + x_after)
                y.append(target_dict[key][i])

            res = np.array(x), np.array(y)
            if train:
                self.train_data = res
            else:
                self.test_data = res
            return res

        return list(map(create_labels_for_set, [True, False]))


def load_data_set():
    ds_location = Path('../data')
    return torch.load(ds_location / 'train_meta.pt'), torch.load(ds_location / 'test_meta.pt')


if __name__ == '__main__':
    smd = SubjectsMetaData('fDemog - Sheet1.csv', '../data/split.json')

    train_dl, test_dl = load_data_set()
    net = torch.load('../sqeuence_last_run.pt')
    svr_loss = []
    for i in [0, 1]:
        (train_x, train_y), (test_x, test_y) = smd.create_labels(i)
        model = SVR(gamma='auto')
        model.fit(train_x, train_y)
        y_hat = model.predict(test_x)
        svr_loss.append(np.mean((test_y - y_hat) ** 2))

        emb_reg = BaseRegression(train_dl, test_dl, net.rnn.initilaizer, {**smd.train_dict, **smd.test_dict}, 3)
        emb_reg.fit(500, i)

    print(f'svr_loss:{svr_loss}')
