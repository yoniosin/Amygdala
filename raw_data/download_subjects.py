from pathlib import Path
import re
from util.config import load_mat
from util.Subject import HealthySubject, PTSDSubject, SubjectMetaData
import numpy as np
from util.config import ROIData
import nibabel as nib
import pickle
import mat4py
from argparse import ArgumentParser


def calc_roi():
    """Load Amygdala ROI matrix, and calculate real voxels and cubic range in each dimension"""
    img = nib.load('PTSD/rrAmygd_ptsd.nii')
    roi = np.where(np.array(img.dataobj))
    amyg_vox = [vox for vox in zip(*roi)]
    min_sizes = map(min, roi)
    max_sizes = map(max, roi)
    h, w, d = list(map(lambda small, big: list(range(small, big + 1)), min_sizes, max_sizes))

    roi_dict = ROIData(amyg_vox, h, w, d)
    return roi_dict


def create_subject(subject_path: Path, subject_type):
    regex = re.search(r'(sub-(\d{3})_ses-TP(\d).*mri(\D*)_bold)\.mat', str(subject_path))
    sub_name, sub_num, session_num, session_type = regex.groups()
    if Path(f'../data/PTSD/PTSD_subject_{sub_num}.pkl').exists() or session_num != '2' or session_type == 'practice':
        return

    subject_idx = protocol['filename'].index(sub_name)
    md = SubjectMetaData(subject_name=sub_num,
                         watch_on=protocol['WatchOnset'][subject_idx],
                         watch_duration=protocol['WatchDuration'][subject_idx],
                         regulate_on=protocol['RegulateOnset'][subject_idx],
                         regulate_duration=protocol['RegulateDuration'][subject_idx],
                         initial_delay=2,
                         )
    try:
        bold_mat = np.array(load_mat(str(subject_path)), dtype=float)
        sub = subject_type(md, bold_mat)
        sub.meta_data = sub.meta_data
        pickle.dump(sub, open(f'../data/PTSD/PTSD_subject_{sub_num}.pkl', 'wb'))
        print(f'Successfully created subject #{sub_num}')
    except:
        print(f'Failed creating subject #{sub_num}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('type', type=str, choices=['healthy', 'PTSD'])

    args = parser.parse_args()
    sub_type = HealthySubject if args.type == 'healthy' else PTSDSubject
    protocol = mat4py.loadmat('PTSD/Protocol.mat')['U']
    roi_dict_path = 'roi_dict.json'
    sub_type.voxels_md = calc_roi()
    raw_data_path = Path('PTSD/DataMat')
    for sub_path in raw_data_path.iterdir():
        create_subject(sub_path, sub_type)
