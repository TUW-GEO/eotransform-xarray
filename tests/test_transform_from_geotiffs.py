import ast
from datetime import datetime
from operator import lt, gt

import numpy as np
import pandas as pd
import rioxarray  # noqa # pylint: disable=unused-import
from eotransform_pandas.transformers.group_by_n import GroupColumnByN
from xarray import DataArray

from assertions import assert_memory_ratio, assert_data_array_identical
from eotransform_xarray.transformers.files_to_xarray import FileDataFrameToDataArray, CONCATED_ATTRS_KEY, BAND_ATTRS_KEY
from factories import make_raster, iota_arrays, generate_yeoda_geo_tiffs
from utils import force_loading, consume


def test_stack_geo_tif_file_dataset_based_on_index(tmp_path):
    times = pd.date_range(datetime(2015, 1, 1, 12, 30, 42), periods=2, freq='D')
    arrays = list(iota_arrays(0, periods=2, shape=(1, 8, 8)))
    geo_tiffs = generate_yeoda_geo_tiffs(tmp_path, times, arrays)
    registered_attribute_parsers = dict(light_direction=ast.literal_eval)

    stacked_array = FileDataFrameToDataArray(registered_attribute_parsers)(geo_tiffs)
    assert_data_array_identical(stacked_array, make_raster(
        np.stack(arrays), dims=['datetime_1', 'band', 'y', 'x'],
        coords=dict(
            datetime_1=DataArray(times, dims=['datetime_1'], attrs={CONCATED_ATTRS_KEY: [{}, {}]}),
            band=[1],
            y=np.arange(8, dtype=np.float),
            x=np.arange(8, dtype=np.float),
            spatial_ref=DataArray(0, attrs=dict(GeoTransform="-0.5 1.0 0.0 -0.5 0.0 1.0"))
        ),
        attrs={CONCATED_ATTRS_KEY: [
            dict(long_name="iota_0", scale_factor=1.0, add_offset=0.0, tags=dict(light_direction=[1, 1, 1])),
            dict(long_name="iota_1", scale_factor=1.0, add_offset=0.0, tags=dict(light_direction=[1, 1, 1]))
        ]}))


def test_stacked_arrays_are_loaded_lazily(tmp_path, disabled_gc):
    times = pd.date_range(datetime(2015, 1, 1, 12, 30, 42), periods=64, freq='D')
    arrays = list(iota_arrays(0, periods=64, shape=(1024, 1024)))
    geo_tiffs = generate_yeoda_geo_tiffs(tmp_path, times, arrays)
    with assert_memory_ratio(1.05, lt):
        stacked_array = FileDataFrameToDataArray()(geo_tiffs)
    with assert_memory_ratio(1.1, gt):
        token = force_loading(stacked_array)
    consume(token)


def test_multi_band_from_multiple_geo_tiffs(tmp_path):
    times = pd.date_range(datetime(2015, 1, 1, 12, 30, 42), periods=4, freq='D')
    arrays = list(iota_arrays(0, periods=4, shape=(1, 8, 8)))
    geo_tiffs = GroupColumnByN('filepath', 2)(generate_yeoda_geo_tiffs(tmp_path, times, arrays))
    registered_attribute_parsers = dict(light_direction=ast.literal_eval)

    stacked_array = FileDataFrameToDataArray(registered_attribute_parsers)(geo_tiffs)
    assert_data_array_identical(stacked_array, make_raster(
        np.stack(arrays).reshape((2, 2, 8, 8)), dims=['datetime_1', 'band', 'y', 'x'],
        coords=dict(
            datetime_1=DataArray(times[::2], dims=['datetime_1'], attrs={CONCATED_ATTRS_KEY: [{}, {}]}),
            band=[0, 1],
            y=np.arange(8, dtype=np.float),
            x=np.arange(8, dtype=np.float),
            spatial_ref=DataArray(0, attrs=dict(GeoTransform="-0.5 1.0 0.0 -0.5 0.0 1.0")),
        ),
        attrs={CONCATED_ATTRS_KEY: [
            {BAND_ATTRS_KEY: [
                dict(long_name="iota_0", scale_factor=1.0, add_offset=0.0, tags=dict(light_direction=[1, 1, 1])),
                dict(long_name="iota_1", scale_factor=1.0, add_offset=0.0, tags=dict(light_direction=[1, 1, 1])),
            ]},
            {BAND_ATTRS_KEY: [
                dict(long_name="iota_2", scale_factor=1.0, add_offset=0.0, tags=dict(light_direction=[1, 1, 1])),
                dict(long_name="iota_3", scale_factor=1.0, add_offset=0.0, tags=dict(light_direction=[1, 1, 1])),
            ]}
        ]}))
