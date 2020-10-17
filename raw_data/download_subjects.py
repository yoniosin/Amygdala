from pathlib import Path
import re
from util.config import load_mat
from data.Subject import Subject, SubjectMetaData
import numpy as np
from util.config import ROIData
import nibabel as nib
import pickle
import mat4py
from argparse import ArgumentParser


def calc_roi(dir_path):
    """Load Amygdala ROI matrix, and calculate real voxels and cubic range in each dimension"""
    img = nib.load(f'{dir_path}/rrAmygd_ptsd.nii')
    roi = np.where(np.array(img.dataobj))
    amyg_vox = [vox for vox in zip(*roi)]
    min_sizes = map(min, roi)
    max_sizes = map(max, roi)
    h, w, d = list(map(lambda small, big: list(range(small, big + 1)), min_sizes, max_sizes))

    return ROIData(amyg_vox, h, w, d)


def create_subject(subject_path: Path, subject_type):
    ptsd_template = r'(sub-(\d{3})_ses-TP(\d).*mri(\D*)_bold)\.mat'
    fibro_template = r'(sub-(\d{3,})-ses-(\d)-\D*(\d{2}))'
    template = ptsd_template if args.type == 'PTSD' else fibro_template
    regex = re.search(template, str(subject_path))
    sub_name, sub_num, session_num, session_type = regex.groups()
    destination_path = Path(f'../data/{args.type}/{args.type}_subject_{sub_num}.pkl')
    if destination_path.exists() or session_num != '2' or session_type == '01':
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
        sub = Subject(md, bold_mat, subject_type=subject_type)
        pickle.dump(sub, open(str(destination_path), 'wb'))
        print(f'Successfully created subject #{sub_num}')
    except:
        print(f'Failed creating subject #{sub_num}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('type', type=str, choices=['healthy', 'PTSD', 'Fibro'])

    args = parser.parse_args()
    protocol = mat4py.loadmat(f'{args.type}/Protocol.mat')['U']
    roi_path = Path(f'{args.type}/roi_dict.pkl')
    if roi_path.exists():
        roi_dict = pickle.load(open(str(roi_path), 'rb'))
    else:
        roi_dict = calc_roi(args.type)
        pickle.dump(roi_dict, open(f'{args.type}/roi_dict.pkl', 'wb'))

    Subject.voxels_md = roi_dict

    raw_data_path = Path(f'{args.type}/DataMat')
    for sub_path in raw_data_path.iterdir():
        create_subject(sub_path, args.type)
