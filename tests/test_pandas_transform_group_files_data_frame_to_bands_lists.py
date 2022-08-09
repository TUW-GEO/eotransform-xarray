from datetime import datetime

import pandas as pd
import pytest

from assertions import assert_data_frame_eq
from eotransform_xarray.transformers.files_data_frame import GroupToBands
from factories import make_files_dataset


def test_group_filepaths_to_bands():
    file_dataset = make_files_dataset(index='datetime_1',
                                      datetime_1=pd.date_range(datetime(2000, 1, 1), datetime(2000, 1, 6)),
                                      filepath=['0.tif', '1.tif', '2.tif', '3.tif', '4.tif', '5.tif'])
    assert_data_frame_eq(GroupToBands(num_bands=3)(file_dataset),
                         make_files_dataset(index='datetime_1',
                                            datetime_1=[datetime(2000, 1, 1), datetime(2000, 1, 4)],
                                            filepaths=[['0.tif', '1.tif', '2.tif'], ['3.tif', '4.tif', '5.tif']]))


def test_error_if_number_of_input_files_is_not_multiple_of_requested_bands():
    file_dataset = make_files_dataset(index='datetime_1',
                                      datetime_1=pd.date_range(datetime(2000, 1, 1), datetime(2000, 1, 5)),
                                      filepath=['0.tif', '1.tif', '2.tif', '3.tif', '4.tif'])
    with pytest.raises(GroupToBands.NumInputFilesMismatchError):
        GroupToBands(num_bands=3)(file_dataset)
