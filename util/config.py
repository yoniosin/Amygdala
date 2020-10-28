from dataclasses import dataclass, asdict, field
from typing import List, Literal, Tuple


# def get_roi_md(dict_path=Path('../PreProcess/voxels_dict.txt'), raw_roi_path='../raw_data/ROI.mat'):
#     if dict_path.exists():
#         return ROIData(**json.load(open(str(dict_path), 'r')))
#
#     roi = np.where(load_mat(str(raw_roi_path)))
#     amyg_vox = [(int(vox[0]), int(vox[1]), int(vox[2])) for vox in zip(*roi)]
#     min_h, min_w, min_d = list(map(min, roi))
#     max_h, max_w, max_d = list(map(max, roi))
#     h_range = list(range(min_h, max_h + 1))
#     w_range = list(range(min_w, max_w + 1))
#     d_range = list(range(min_d, max_d + 1))
#
#     roi_dict = ROIData(amyg_vox, h_range, w_range, d_range)
#     json.dump(asdict(roi_dict), open(str(dict_path), 'w'))
#     return roi_dict


@dataclass
class ROIData:
    amyg_vox: List
    h_range: List
    w_range: List
    d_range: List


@dataclass
class LearnerConfig:
    max_epochs: int = 500
    batch_size: int = 10
    train_ratio: float = 0.8
    train_windows: int = 2
    runs_dir: str = 'C:/Users/yonio/PycharmProjects/Amygdala_new/runs'
    main_dir: str = 'C:/Users/yonio/PycharmProjects/Amygdala_new'

    def validate_config(self):
        assert 0 < self.train_ratio < 1


@dataclass
class fMRILearnerConfig(LearnerConfig):
    use_embeddings: str = None
    min_w: int = field(init=False)
    voxels_num: int = field(init=False)
    in_channels: int = field(init=False)

    def update_cfg(self):
        # meta_dict = json.load(open('meta.txt', 'r'))
        # self.total_subject = 60
        self.min_w = 14
        # self.voxels_num = meta_dict['voxels_num']
        self.in_channels = self.train_windows * 2 + 1
        self.logger_path = f'{self.runs_dir}/run#{self.run_num}({self.use_embeddings})'

    def to_json(self): return asdict(self)

    def validate_config(self):
        super().validate_config()
        assert 0 < self.train_windows < 5


@dataclass
class EEGNetConfig:
    watch_hidden_size: int = 13
    reg_hidden_size: int = 13
    embedding_size: int = 5
    watch_len: int = 60
    reg_len: int = 180
    lr: float = 1e-2
    weight_decay: float = 0
    n_subjects: int = 164
    train_lstm: bool = True


@dataclass
class DataPaths:
    type: str
    eeg_dir: str
    criteria_dir: str


@dataclass
class EEGData:
    db_type: Tuple[str] = ('ptsd',)
    load: bool = True
    re_split: bool = False
    use_criteria: bool = False
    criteria_len: int = 3
    ptsd_paths: DataPaths = DataPaths(
        'ptsd',
        '../../../data/eeg/processed/PTSD',
        r'C:\Users\yonio\PycharmProjects\Amygdala_new\MetaData\PTSD\Clinical.csv')

    control_paths: DataPaths = DataPaths(
        'control',
        '../Amygdala/data/3D',
        '../Amygdala/MetaData/fDemog.csv'
    )

    fibro_paths: DataPaths = DataPaths(
        'fibro',
        'data/Fibro',
        'MetaData/Fibro/Clinical.csv'
    )


@dataclass
class EEGLearnerConfig:
    full_train: bool = False
    validation: bool = False
    learner: LearnerConfig = LearnerConfig()
    net: EEGNetConfig = EEGNetConfig()
    data: EEGData = EEGData()


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
