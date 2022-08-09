from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Callable, Optional, Sequence, AnyStr, Pattern

import pandas as pd
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename


def yeoda_naming_convention(file_name: str) -> Dict:
    try:
        yeoda_file = YeodaFilename.from_filename(file_name, convert=True)
        return {k: yeoda_file[k] for k in yeoda_file.fields_def if yeoda_file[k]}
    except ValueError:
        return {}


def gather_files(root: Path, naming_convention: Callable[[str], Dict],
                 sub_folder_structure: Optional[Sequence[Pattern[AnyStr]]] = None,
                 index: Optional[str] = None) -> pd.DataFrame:
    directories = list()
    _add_sub_folders(root, deque(sub_folder_structure or []), directories)

    files = [(file, naming_convention(file.name)) for directory in directories for file in directory.glob("*.*")]
    data = defaultdict(list)
    for file, meta_data in files:
        if meta_data:
            _add_file_and_meta_data(data, file, meta_data)

    return _make_data_frame_from(data, index)


def _add_sub_folders(current: Path, sub_folders: deque, file_list: list):
    if sub_folders:
        sub_pattern = sub_folders.popleft()
        for directory in current.iterdir():
            if directory.is_dir() and sub_pattern.fullmatch(directory.name):
                _add_sub_folders(directory, deque(sub_folders), file_list)
    else:
        file_list.append(current)


def _make_data_frame_from(data, index):
    df = pd.DataFrame(data)
    if index:
        df.set_index(index, inplace=True)
        df.sort_index(inplace=True)
    return df


def _add_file_and_meta_data(data, file, meta_data):
    data['filepath'].append(file)
    for k, v in meta_data.items():
        data[k].append(v)
