from util.config import SubjectMetaData
from itertools import chain
from abc import abstractmethod
import torch
import numpy as np


class Subject:
    voxels_md = None

    def __init__(self, meta_data: SubjectMetaData, bold_mat, window_data_type):
        def gen_windows(window_type):
            times_list = self.meta_data.watch_times if window_type == 'watch' else self.meta_data.regulate_times
            return map(lambda w: window_data_type(*w, window_type, bold_mat), enumerate(times_list))

        self.meta_data = meta_data
        self.name = meta_data.subject_name
        self.paired_windows = list(map(PairedWindows, gen_windows('watch'), gen_windows('regulate')))

    def get_data(self, train_num, width, scalar_result=True):
        if scalar_result:
            prev_data = list(chain(*[w.get_data(width) for w in self.get_windows(train_num)]))

            last_pw = self.paired_windows[train_num]
            last_data = last_pw.get_data(width)
            X = np.hstack(prev_data + last_data)
            y = last_pw.score
            return X, y
        else:
            res = torch.stack([w.get_data(width) for w in self.get_windows(train_num + 1)])
            return res

    def get_single_experience(self, idx, width):
        return self.paired_windows[idx].get_data(width)

    def __repr__(self):
        grades = [pw.score for pw in self.paired_windows]
        grades_formatted = ("{:.2f}, " * len(grades)).format(*grades)
        return f'{self.name} windows grades=[{grades_formatted}]'

    def get_windows(self, windows_num): return self.paired_windows[:windows_num]

    def __len__(self):
        return len(self.paired_windows)


class PairedWindows:
    def __init__(self, watch_window, regulate_window):
        def calc_score():
            mean_diff = self.watch_window.mean - self.regulate_window.mean
            joint_var = torch.var(torch.cat((self.watch_window.bold, self.regulate_window.bold), dim=3))
            return mean_diff / joint_var

        assert watch_window.idx == regulate_window.idx, f'indices mismatch: {watch_window.idx} != {regulate_window.idx}'
        self.idx = watch_window.idx
        self.watch_window: Window = watch_window
        self.regulate_window: Window = regulate_window
        self.score = calc_score()

    def __repr__(self):
        return f'Windows #{self.idx}, score = {self.score:.4f}'

    def get_data(self, width):
        res = torch.stack([w.get_data(width) for w in (self.watch_window, self.regulate_window)])
        return res


class Window:
    def __init__(self, idx, time_slots, window_type, bold_mat):
        self.idx = idx
        self.time = time_slots
        self.window_type = window_type
        self.bold = self.gen_bold_mat(bold_mat)

    @abstractmethod
    def gen_bold_mat(self, *args): pass

    @abstractmethod
    def get_data(self, width): pass

    @property
    def mean(self): return torch.mean(self.bold)


class FlatWindow(Window):
    def __init__(self, idx, time_slots, window_type, bold_mat):
        class Voxel:
            def __init__(self, vox_coor, samples):
                self.vox_coor = vox_coor
                self.samples = samples

            def __repr__(self):
                return f"vox {self.vox_coor}"
        self.voxels = {vox: Voxel(vox, bold_mat[(*vox), time_slots]) for vox in Subject.voxels_md.amyg_vox}
        super().__init__(idx, time_slots, window_type, bold_mat)

    def gen_bold_mat(self, bold_mat):
        return torch.tensor([voxel.samples for voxel in self.voxels.values()])

    def get_data(self, width): return self.bold[:, :width]


class Window3D(Window):
    def __init__(self, idx, time_slots, window_type, bold_mat):
        super().__init__(idx, time_slots, window_type, bold_mat)

    def gen_bold_mat(self, bold_mat):
        x = bold_mat[Subject.voxels_md.h_range, :, :, :]
        x = x[:, Subject.voxels_md.w_range, :, :]
        x = x[:, :, Subject.voxels_md.d_range, :]
        return torch.tensor(x[:, :, :, self.time])

    def get_data(self, width): return self.bold[:, :, :, :width]


def subject_generator(subject_id, protocol, bold_mat, data_type='3d'):
    md = SubjectMetaData(subject_id, *protocol)
    window_data_type = Window3D if data_type == '3d' else Window
    return Subject(meta_data=md, bold_mat=bold_mat, window_data_type=window_data_type)
