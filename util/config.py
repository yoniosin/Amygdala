from dataclasses import dataclass, asdict, field
from typing import List
import json
import numpy as np
from oct2py import Oct2Py
from pathlib import Path


def load_mat(path):
    oc = Oct2Py()
    ans = oc.convert_img(path)
    oc.exit()
    return ans


def get_roi_md(dict_path=Path('../PreProcess/voxels_dict.txt'), raw_roi_path='../raw_data/ROI.mat'):
    if dict_path.exists():
        return ROIData(**json.load(open(str(dict_path), 'r')))

    roi = np.where(load_mat(str(raw_roi_path)))
    amyg_vox = [(int(vox[0]), int(vox[1]), int(vox[2])) for vox in zip(*roi)]
    min_h, min_w, min_d = list(map(min, roi))
    max_h, max_w, max_d = list(map(max, roi))
    h_range = list(range(min_h, max_h + 1))
    w_range = list(range(min_w, max_w + 1))
    d_range = list(range(min_d, max_d + 1))

    roi_dict = ROIData(amyg_vox, h_range, w_range, d_range)
    json.dump(asdict(roi_dict), open(str(dict_path), 'w'))
    return roi_dict


@dataclass
class ROIData:
    amyg_vox: List
    h_range: List
    w_range: List
    d_range: List


@dataclass
class LearnerConfig:
    run_num: int
    batch_size: int = 2
    train_ratio: float = 0.8
    train_windows: int = 2
    total_subject: int = field(init=False)
    logger_path: str = field(init=False)
    runs_dir: str = 'runs'

    def __post_init__(self):
        self.validate_config()

    def validate_config(self):
        assert 0 < self.train_ratio < 1


@dataclass
class fMRILearnerConfig(LearnerConfig):
    use_embeddings: str = None
    min_w: int = field(init=False)
    voxels_num: int = field(init=False)
    in_channels: int = field(init=False)

    def __post_init__(self):
        # meta_dict = json.load(open('meta.txt', 'r'))
        # self.total_subject = 60
        super().__post_init__()
        self.min_w = 14
        # self.voxels_num = meta_dict['voxels_num']
        self.in_channels = self.train_windows * 2 + 1
        self.logger_path = f'{self.runs_dir}/run#{self.run_num}({self.use_embeddings})'

    def to_json(self): return asdict(self)

    def validate_config(self):
        super().validate_config()
        assert 0 < self.train_windows < 5


@dataclass
class EEGLearnerConfig(LearnerConfig):
    def __post_init__(self):
        super().__post_init__()
        self.logger_path = f'{self.runs_dir}/eeg/run#{self.run_num}'


@dataclass
class SubjectMetaData:
    subject_name: str
    watch_on: List[int]
    watch_duration: List[int]
    regulate_on: List[int]
    regulate_duration: List[int]
    initial_delay: int = 2
    subject_type: str = 'healthy'

    def gen_time_range(self, on, duration): return list(range(on + self.initial_delay, on + duration))

    def __post_init__(self):
        self.min_w = min(self.watch_duration + self.regulate_duration) - self.initial_delay
        self.watch_times = map(self.gen_time_range, self.watch_on, self.watch_duration)
        self.regulate_times = map(self.gen_time_range, self.regulate_on, self.regulate_duration)


if __name__ == '__main__':
    get_roi_md(Path(r'rawData/voxels_dict.json'), Path(r'raw_data/rrAmygd_ptsd.nii'))