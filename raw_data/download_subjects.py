import json
from pathlib import Path
import re
from util.config import load_mat
from util.Subject import Subject, SubjectMetaData


if __name__ == '__main__':
    raw_data_path = Path('DataMat')
    for sub_path in raw_data_path.iterdir():
        sub_num, session_num = re.search(r'sub-(\d{3})_ses-TP(\d)', str(sub_path)).groups()
        if session_num != '1':
            continue

        bold_mat = load_mat(str(sub_path))
        md = SubjectMetaData(subject_name='',
                             watch_on=[0],
                             watch_duration=[0],
                             regulate_on=[0],
                             regulate_duration=[0]
                             )
        sub = Subject(md, bold_mat)
        json.dump(sub, open(f'subject_{sub_num}.json', 'wb'))

