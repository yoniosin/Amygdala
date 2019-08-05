from pathlib import Path
import pickle

if __name__ == '__main__':
    subjects_dir_path = Path('/home/yonio/Projects/conv_gru/3d_data/3D')
    subjects_score = {}

    for subject_path in subjects_dir_path.iterdir():
        subject = pickle.load(open(str(subject_path), 'rb'))
        subjects_score[subject.name] = subject.score