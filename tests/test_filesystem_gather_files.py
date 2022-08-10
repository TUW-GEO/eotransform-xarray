import re
from datetime import datetime

import pandas as pd
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename

from assertions import assert_data_frame_eq
from eotransform_xarray.filesystem.gather import gather_files, yeoda_naming_convention
from factories import make_files_dataset

DAY_0 = datetime(2015, 1, 1, 14, 30, 21)
DAY_1 = datetime(2015, 1, 2, 14, 30, 21)
DAY_2 = datetime(2015, 1, 3, 14, 30, 21)
DAY_3 = datetime(2015, 1, 4, 14, 30, 21)
DAY_4 = datetime(2015, 1, 5, 14, 30, 21)


def test_gather_files_based_on_naming_convention(tmp_path):
    file_0 = touch_yeoda_file_at(tmp_path, datetime_1=DAY_0, var_name="TEST")
    file_1 = touch_yeoda_file_at(tmp_path, datetime_1=DAY_1, var_name="TEST")
    files_dataset = gather_files(tmp_path, yeoda_naming_convention, index='datetime_1')
    assert_data_frame_eq(files_dataset, make_files_dataset(index="datetime_1",
                                                           datetime_1=pd.to_datetime([DAY_0, DAY_1]),
                                                           filepath=[file_0, file_1],
                                                           var_name=["TEST", "TEST"]))


def touch_yeoda_file_at(path, **attributes):
    yeoda_name = str(YeodaFilename(attributes, convert=True))
    file = path / yeoda_name
    file.parent.mkdir(parents=True, exist_ok=True)
    file.touch()
    return file


def test_ignore_files_not_following_naming_convention(tmp_path):
    random_file = (tmp_path / "some_file.txt")
    random_file.touch()
    file_0 = touch_yeoda_file_at(tmp_path, datetime_1=DAY_0)
    files_dataset = gather_files(tmp_path, yeoda_naming_convention, index='datetime_1')
    assert_data_frame_eq(files_dataset, make_files_dataset(index="datetime_1",
                                                           datetime_1=pd.to_datetime([DAY_0]),
                                                           filepath=[file_0]))


def test_search_files_in_sub_folder_structure(tmp_path):
    file_0 = touch_yeoda_file_at(tmp_path / "grid0/tile0", datetime_1=DAY_0)
    file_1 = touch_yeoda_file_at(tmp_path / "grid0/tile1", datetime_1=DAY_1)
    file_2 = touch_yeoda_file_at(tmp_path / "grid1/tile0", datetime_1=DAY_2)
    touch_yeoda_file_at(tmp_path / "invalid_grid/tile0", datetime_1=DAY_3)
    touch_yeoda_file_at(tmp_path / "grid1/invalid_tile", datetime_1=DAY_4)

    files_dataset = gather_files(tmp_path, yeoda_naming_convention,
                                 sub_folder_structure=[re.compile(r"grid\d"), re.compile(r"tile\d")],
                                 index='datetime_1')
    assert_data_frame_eq(files_dataset, make_files_dataset(index="datetime_1",
                                                           datetime_1=pd.to_datetime([DAY_0, DAY_1, DAY_2]),
                                                           filepath=[file_0, file_1, file_2]))
