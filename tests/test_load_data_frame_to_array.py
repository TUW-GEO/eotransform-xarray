from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from xarray import DataArray

from assertions import assert_data_array_identical
from eotransform_xarray.functional.load_file_dataframe_to_array import load_file_dataframe_to_array, CONCATED_ATTRS_KEY
from factories import iota_arrays, generate_yeoda_geo_tiffs, make_raster


def test_loads_data_frame_according_to_index(tmp_path):
    times = pd.date_range(datetime(2015, 1, 1, 12, 30, 42), periods=2, freq='D')
    arrays = list(iota_arrays(0, periods=2, shape=(1, 8, 8)))
    geo_tiffs = generate_yeoda_geo_tiffs(tmp_path, times, arrays)
    files = geo_tiffs['filepath'].tolist()

    stacked_array = load_file_dataframe_to_array(geo_tiffs)

    assert_data_array_identical(stacked_array, make_raster(
        np.stack(arrays), dims=['datetime_1', 'band', 'y', 'x'],
        coords=dict(
            datetime_1=DataArray(times, dims=['datetime_1'], attrs={CONCATED_ATTRS_KEY: [{}, {}]}),
            filepath=DataArray(files, dims=['datetime_1'], attrs={CONCATED_ATTRS_KEY: [{}, {}]}),
            band=[1],
            y=np.arange(8, dtype=np.float),
            x=np.arange(8, dtype=np.float),
            spatial_ref=DataArray(0, attrs=dict(GeoTransform="-0.5 1.0 0.0 -0.5 0.0 1.0"))
        ),
        attrs={CONCATED_ATTRS_KEY: [
            {'scale_factor': 1.0, 'add_offset': 0.0, 'long_name': "iota_0"},
            {'scale_factor': 1.0, 'add_offset': 0.0, 'long_name': "iota_1"}
        ]}))


@pytest.mark.parametrize('legacy_scale_factor', [
    {'Scale_factor': '100'}, {'scale_factor': '100'}])
def test_load_arrays_with_legacy_meta_data(tmp_path, legacy_scale_factor):
    geo_tiffs = generate_yeoda_geo_tiffs(tmp_path, [datetime(2015, 1, 1, 12, 30, 42)], [[[10]]],
                                         attrs=legacy_scale_factor, legacy=True)
    scaled_value = load_file_dataframe_to_array(geo_tiffs, allow_legacy_scaling=True).squeeze().values.item()
    assert scaled_value == 0.1
