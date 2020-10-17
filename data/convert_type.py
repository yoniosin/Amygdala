from pathlib import Path
from data.Subject import Subject
import pickle

if __name__ == '__main__':
    for subject_path in Path('.').iterdir():
        if subject_path.name.endswith('py'): continue
        subject: Subject = pickle.load(open(str(subject_path), 'rb'))
        subject.convert_to_fibro()
        pickle.dump(subject, open(str(subject_path), 'wb'))
