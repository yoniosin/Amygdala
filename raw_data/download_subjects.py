from pathlib import Path
import re
from util.config import load_mat
from util.Subject import Subject, SubjectMetaData
import numpy as np
from util.config import ROIData
import nibabel as nib
import pickle
import mat4py


def calc_roi():
    """Load Amygdala ROI matrix, and calculate real voxels and cubic range in each dimension"""
    img = nib.load('rrAmygd_ptsd.nii')
    roi = np.where(np.array(img.dataobj))
    amyg_vox = [vox for vox in zip(*roi)]
    min_sizes = map(min, roi)
    max_sizes = map(max, roi)
    h, w, d = list(map(lambda small, big: list(range(small, big + 1)), min_sizes, max_sizes))

    roi_dict = ROIData(amyg_vox, h, w, d)
    return roi_dict


if __name__ == '__main__':
    protocol = mat4py.loadmat('Protocol.mat')
    roi_dict_path = 'roi_dict.json'
    Subject.voxels_md = calc_roi()
    raw_data_path = Path('DataMat')
    for sub_path in raw_data_path.iterdir():
        sub_num, session_num = re.search(r'sub-(\d{3})_ses-TP(\d)', str(sub_path)).groups()
        if session_num != '1':
            continue

        bold_mat = np.array(load_mat(str(sub_path)), dtype=float)
        md = SubjectMetaData(subject_name=sub_num,
                             watch_on=[0],
                             watch_duration=[124],
                             regulate_on=[0],
                             regulate_duration=[124],
                             initial_delay=2,
                             )
        sub = Subject(md, bold_mat)
        sub.meta_data = sub.meta_data
        pickle.dump(sub, open(f'subject_{sub_num}.pkl', 'wb'))
