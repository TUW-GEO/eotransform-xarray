import ast
from copy import deepcopy
from datetime import datetime
from operator import lt, gt

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa # pylint: disable=unused-import
from eotransform_pandas.transformers.group_by_n import GroupColumnByN
from xarray import DataArray

from assertions import assert_memory_ratio, assert_data_array_identical
from eotransform_xarray.functional.load_file_dataframe_to_array import CONCATED_ATTRS_KEY, BAND_ATTRS_KEY
from eotransform_xarray.transformers.files_to_xarray import FileDataFrameToDataArray
from factories import make_raster, iota_arrays, generate_yeoda_geo_tiffs
from utils import force_loading, consume


class FunctionSpy:
    def __init__(self, wrapped):
        self._wrapped = wrapped
        self.received_args = []
        self.received_kwargs = []

    def __call__(self, *args, **kwargs):
        self.received_args.append(args)
        self.received_kwargs.append(deepcopy(kwargs))
        return self._wrapped(*args, **kwargs)


@pytest.fixture
def rioxarray_rasterio_open_spy(monkeypatch):
    with monkeypatch.context() as m:
        spy = FunctionSpy(rioxarray.open_rasterio)
        m.setattr(rioxarray, "open_rasterio", spy)
        yield spy


@pytest.fixture
def rasterio_open_spy(monkeypatch):
    from eotransform_xarray.functional.load_tif import rasterio
    with monkeypatch.context() as m:
        spy = FunctionSpy(rasterio.open)
        m.setattr(rasterio, "open", spy)
        yield spy


def test_stack_geo_tif_file_dataset_based_on_index(tmp_path):
    times = pd.date_range(datetime(2015, 1, 1, 12, 30, 42), periods=2, freq='D')
    arrays = list(iota_arrays(0, periods=2, shape=(1, 8, 8)))
    geo_tiffs = generate_yeoda_geo_tiffs(tmp_path, times, arrays, attrs=dict(light_direction=[1, 1, 1]))
    files = geo_tiffs['filepath'].tolist()

    registered_attribute_parsers = dict(light_direction=ast.literal_eval)

    stacked_array = FileDataFrameToDataArray(registered_attribute_parsers)(geo_tiffs)
    assert_data_array_identical(stacked_array, make_raster(
        np.stack(arrays), dims=['datetime_1', 'band', 'y', 'x'],
        coords=dict(
            datetime_1=DataArray(times, dims=['datetime_1'], attrs={CONCATED_ATTRS_KEY: [{}, {}]}),
            filepath=DataArray(files, dims=['datetime_1'], attrs={CONCATED_ATTRS_KEY: [{}, {}]}),
            band=[1],
            y=np.arange(8, dtype=np.float64),
            x=np.arange(8, dtype=np.float64),
            spatial_ref=DataArray(0, attrs=dict(GeoTransform="-0.5 1.0 0.0 -0.5 0.0 1.0"))
        ),
        attrs={CONCATED_ATTRS_KEY: [
            {'long_name': "iota_0", 'scale_factor': 1.0, 'add_offset': 0.0, 'light_direction': [1, 1, 1]},
            {'long_name': "iota_1", 'scale_factor': 1.0, 'add_offset': 0.0, 'light_direction': [1, 1, 1]},
        ]}))


def test_stacked_arrays_are_loaded_lazily(tmp_path, disabled_gc):
    times = pd.date_range(datetime(2015, 1, 1, 12, 30, 42), periods=64, freq='D')
    arrays = list(iota_arrays(0, periods=64, shape=(1024, 1024)))
    geo_tiffs = generate_yeoda_geo_tiffs(tmp_path, times, arrays)
    with assert_memory_ratio(1.05, lt):
        stacked_array = FileDataFrameToDataArray(rioxarray_kwargs=dict(chunks=True))(geo_tiffs)
    with assert_memory_ratio(1.1, gt):
        token = force_loading(stacked_array)
    consume(token)


def test_multi_band_from_multiple_geo_tiffs(tmp_path):
    times = pd.date_range(datetime(2015, 1, 1, 12, 30, 42), periods=4, freq='D')
    arrays = list(iota_arrays(0, periods=4, shape=(1, 8, 8)))
    geo_tiffs = GroupColumnByN('filepath', 2)(generate_yeoda_geo_tiffs(tmp_path, times, arrays,
                                                                       attrs=dict(light_direction=[1, 1, 1])))
    file_lists = make_file_list_array(geo_tiffs)

    registered_attribute_parsers = dict(light_direction=ast.literal_eval)

    stacked_array = FileDataFrameToDataArray(registered_attribute_parsers)(geo_tiffs)
    assert_data_array_identical(stacked_array, make_raster(
        np.stack(arrays).reshape((2, 2, 8, 8)), dims=['datetime_1', 'band', 'y', 'x'],
        coords=dict(
            datetime_1=DataArray(times[::2], dims=['datetime_1'], attrs={CONCATED_ATTRS_KEY: [{}, {}]}),
            filepaths=DataArray(file_lists, dims=['datetime_1'], attrs={CONCATED_ATTRS_KEY: [{}, {}]}),
            band=[0, 1],
            y=np.arange(8, dtype=np.float64),
            x=np.arange(8, dtype=np.float64),
            spatial_ref=DataArray(0, attrs=dict(GeoTransform="-0.5 1.0 0.0 -0.5 0.0 1.0")),
        ),
        attrs={CONCATED_ATTRS_KEY: [
            {BAND_ATTRS_KEY: [
                {'long_name': "iota_0", 'scale_factor': 1.0, 'add_offset': 0.0, 'light_direction': [1, 1, 1]},
                {'long_name': "iota_1", 'scale_factor': 1.0, 'add_offset': 0.0, 'light_direction': [1, 1, 1]},
            ]},
            {BAND_ATTRS_KEY: [
                {'long_name': "iota_2", 'scale_factor': 1.0, 'add_offset': 0.0, 'light_direction': [1, 1, 1]},
                {'long_name': "iota_3", 'scale_factor': 1.0, 'add_offset': 0.0, 'light_direction': [1, 1, 1]},
            ]}
        ]}))


def make_file_list_array(geo_tiffs):
    fl_array = np.empty(2, dtype=object)
    file_lists = geo_tiffs['filepaths'].tolist()
    fl_array[0] = file_lists[0]
    fl_array[1] = file_lists[1]
    return fl_array


def test_pass_on_kwargs_to_io_drivers(tmp_path, rioxarray_rasterio_open_spy, rasterio_open_spy):
    times = pd.date_range(datetime(2015, 1, 1, 12, 30, 42), periods=2, freq='D')
    arrays = list(iota_arrays(0, periods=2, shape=(8, 8)))
    geo_tiffs = generate_yeoda_geo_tiffs(tmp_path, times, arrays)
    stacked_array = FileDataFrameToDataArray(rioxarray_kwargs=dict(chunks=(1, 4, 4), sharing=True))(geo_tiffs)
    consume(stacked_array)
    assert rioxarray_rasterio_open_spy.received_kwargs[-1]['chunks'] == (1, 4, 4)
    assert rasterio_open_spy.received_kwargs[-2]['sharing'] == True
