from util.config import EEGLearnerConfig
from util.AmygDataSet import ScoresAmygDataset, EEGDataSet
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import json
from typing import Iterable
from MetaData.SubjectMetaData import FibroSubjectMetaData, PTSDSubjectMetaData, HealthySubjectMetaData
from dataclasses import dataclass
import Models.Predictors as Prd
from util.Subject import EEGSubjectPTSD, PairedWindows, EEGWindow
import hydra
from hydra.core.config_store import ConfigStore


@dataclass
class DBWrapper:
    db_type: str
    data_path: Path
    meta_data_path: str
    split_path: str = 'split.json'

    def generate_meta_data(self, runs_dir: Path):
        if self.db_type == 'healthy':
            meta_data_type = HealthySubjectMetaData
        elif self.db_type == 'PTSD':
            meta_data_type = PTSDSubjectMetaData
        elif self.db_type == 'Fibro':
            meta_data_type = FibroSubjectMetaData
        else:
            raise ValueError(f'Illegal value for Meta data type: {self.db_type}')

        self.meta_data = meta_data_type(runs_dir/self.meta_data_path, runs_dir/self.split_path)


def load_data_set(db_wrapper_list: Iterable[DBWrapper], cfg):
    ds_location = Path('data/eeg')
    if cfg.data.load and (ds_location / 'train.pt').exists() and (ds_location / 'test.pt').exists():
        return torch.load(ds_location / 'train.pt'), torch.load(ds_location / 'test.pt'), torch.load(
            ds_location / 'input_shape.pt')

    ds = EEGDataSet([db.data_path for db in db_wrapper_list], cfg.learner)
    train_ds, test_ds = ds.train_test_split()

    train_dl_ = DataLoader(train_ds, batch_size=cfg.learner.batch_size, shuffle=True)
    torch.save(train_dl_, '../../../data/eeg/train.pt')
    test_dl_ = DataLoader(test_ds, batch_size=cfg.learner.batch_size, shuffle=True)
    torch.save(test_dl_, '../../../data/eeg/test.pt')
    # torch.save(ds.get_sample_shape(), 'data/input_shape.pt')
    return train_dl_, test_dl_, ds.get_sample_shape()


# def get_person_embedding(fc_layer):
#     def create_one_hot(pesron_id):
#         res = torch.zeros(87)
#         res[pesron_id] = 1
#         return res
#
#     persons = torch.stack([create_one_hot(i) for i in range(87)])
#     full_embedding = fc_layer(persons)
#     return list(zip(persons, full_embedding))
#
#
# def varied_sizes_collate(batch):
#     res = defaultdict(list)
#     for subject in batch:
#         for key in subject.keys():
#             res[key].append(subject[key])
#
#     return res
#
#
# def create_mapping(dir_iterator: List[Path]):
#     joint_iter = chain(*[directory.iterdir() for directory in dir_iterator])
#     subjects_dict = {int(re.search(r'(\d*).pkl$', str(path)).group(1)): i for i, path in enumerate(joint_iter)}
#     json.dump(subjects_dict, open('mapping.json', 'w'))


def prepare_run():
    # if args.create_mapping:
    #     create_mapping([Path('../Amygdala/data/3D'), Path('data/PTSD'), Path('data/Fibro')])

    return EEGLearnerConfig(batch_size=10,
                            train_ratio=0.9,
                            train_windows=1
                            )


def upload_db(cfg, runs_path):
    db_list_ = [DBWrapper(db_type, *cfg.data.paths[db_type]) for db_type in cfg.data.db_type]
    train_dl_, test_dl_, input_shape_ = load_data_set(db_list_, cfg)
    # for db in db_list_:
    #     db.generate_meta_data(runs_path)
    return {'train_dl': train_dl_, 'test_dl': test_dl_}


@hydra.main(config_name='eeg')
def main(cfg: EEGLearnerConfig):
    runs_dir = Path(cfg.learner.runs_dir)
    run_num_path = runs_dir/'eeg/last_run.json'
    run_num = json.load(open(str(run_num_path), 'r'))['last'] + 1
    cfg.learner.run_num = run_num
    update_cfg(cfg)

    data_loaders = upload_db(cfg, runs_dir)

    model = Prd.EEGModel(
        **data_loaders,
        **cfg.net,
        run_num=run_num,
        logger_path=cfg.learner.logger_path)
    model.train(cfg.learner.max_epochs)
    json.dump({"last": run_num}, open(str(run_num_path), 'w'))


def update_cfg(cfg: EEGLearnerConfig):
    assert 0 < cfg.learner.train_ratio <= 1
    cfg.learner.logger_path = f'{cfg.learner.runs_dir}/eeg/run#{cfg.learner.run_num}'


if __name__ == '__main__':
    cs = ConfigStore()
    cs.store(name='eeg', node=EEGLearnerConfig)
    main()
