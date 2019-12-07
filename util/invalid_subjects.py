from pathlib import Path
from typing import Iterable
import csv
import json
import re


def update_invalid(invalid_dict_path: Path, md_iter: Iterable[Path]):
    try:
        invalid_set = set(json.load(open(str(invalid_dict_path), 'r')))
    except:
        invalid_set = set()
    for csv_path in md_iter:
        with open(str(csv_path)) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for line in csv_reader:
                if any(map(lambda t: t in['#N/A', ''], (line['TAS'], line['STAI']))):
                    sub_num = re.search(r'(\d+)', line[line.keys().__iter__().__next__()]).group(1)
                    invalid_set.add(int(sub_num))

    json.dump(list(invalid_set), open(str(invalid_dict_path), 'w'))


if __name__ == '__main__':
    update_invalid(Path('../invalid_subjects.json'),
                   [Path('../../Amygdala/MetaData/fDemog.csv'),
                    Path('../MetaData/Fibro/Clinical.csv'),
                    Path('../MetaData/PTSD/Clinical.csv')])
