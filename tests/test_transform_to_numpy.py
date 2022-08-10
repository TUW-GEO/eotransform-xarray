from datetime import datetime

import numpy as np
import pandas as pd

from eotransform_xarray.transformers.to_numpy import NDArrayWithAttributes, NDArrayMapping, DatasetToNumpy, \
    DataArrayToNumpy
from factories import make_raster, make_dataset


def test_transform_data_array_to_numpy_array():
    source = make_raster(np.random.randn(4, 1, 8, 8), dims=['time', 'band', 'y', 'x'], coords=dict(
        time=pd.date_range(datetime(2000, 1, 1), datetime(2000, 1, 4), freq='D'),
        band=[1],
        y=np.linspace(0, 1, 8),
        x=np.linspace(0, 1, 8),
    ), attrs=dict(some_attr=42))
    assert_are_equal_numpy_arrays_with_attrs(DataArrayToNumpy()(source),
                                             NDArrayWithAttributes(source.values, source.attrs))


def assert_are_equal_numpy_arrays_with_attrs(actual, expected):
    assert isinstance(actual, NDArrayWithAttributes)
    np.testing.assert_equal(actual.values, expected.values)
    assert actual.attrs == expected.attrs


def test_transform_data_array_to_numpy_array_mapping():
    raster = make_raster(np.random.randn(4, 1, 8, 8), dims=['time', 'band', 'y', 'x'],
                         coords=dict(time=pd.date_range(datetime(2000, 1, 1), datetime(2000, 1, 4), freq='D'), band=[1],
                                     y=np.linspace(0, 1, 8), x=np.linspace(0, 1, 8), ), attrs=dict(some_attr=42))
    source = make_dataset(dict(raster=raster), attrs=dict(some='dataset attribute'))
    asset_are_equal_numpy_array_mapping(DatasetToNumpy()(source),
                                        NDArrayMapping(
                                            dict(raster=NDArrayWithAttributes(raster.values, raster.attrs)),
                                            attrs=dict(some='dataset attribute'),
                                        ))


def asset_are_equal_numpy_array_mapping(actual, expected):
    assert len(actual) == len(expected)

    for k in actual:
        assert_are_equal_numpy_arrays_with_attrs(actual[k], expected[k])

    assert actual.attrs == expected.attrs
