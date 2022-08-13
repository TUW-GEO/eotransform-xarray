import numpy as np
from xarray import DataArray

from assertions import assert_data_array_eq, assert_dataset_identical
from eotransform_xarray.transformers.reshape_by_coords import ReshapeByCoords
from factories import make_raster, make_dataset


def test_reshape_array_by_coordinates():
    flat_array = make_array([[10, 20, 30, 40],
                             [11, 21, 31, 41]], dims=("time", "locations"),
                            coords=dict(y=('locations', [0, 0, 1, 1]), x=('locations', [-2, -1, -2, -1])))
    raster_array = ReshapeByCoords('locations', ('y', 'x'))(flat_array)
    assert_data_array_eq(raster_array, make_raster([[[10, 20],
                                                     [30, 40]],
                                                    [[11, 21],
                                                     [31, 41]]],
                                                   dims=('time', 'y', 'x'), coords=dict(y=[0, 1], x=[-2, -1])))


def make_array(values, dims, coords):
    return DataArray(np.array(values), dims=dims, coords=coords)


def test_reshape_dataset_by_coordinates():
    flat_foo = make_array([[10, 20, 30, 40],
                           [11, 21, 31, 41]], dims=("time", "locations"),
                          coords=dict(y=('locations', [0, 0, 1, 1]), x=('locations', [-2, -1, -2, -1])))
    flat_bar = make_array([[1.0, 2.0, 3.0, 4.0],
                           [1.1, 2.1, 3.1, 4.1]], dims=("time", "locations"),
                          coords=dict(y=('locations', [0, 0, 1, 1]), x=('locations', [-2, -1, -2, -1])))
    raster_ds = ReshapeByCoords('locations', ('y', 'x'))(make_dataset(dict(foo=flat_foo, bar=flat_bar)))
    assert_dataset_identical(raster_ds, make_dataset(
        dict(foo=make_raster([[[10, 20],
                               [30, 40]],
                              [[11, 21],
                               [31, 41]]],
                             dims=('time', 'y', 'x'), coords=dict(y=[0, 1], x=[-2, -1])),
             bar=make_raster([[[1.0, 2.0],
                               [3.0, 4.0]],
                              [[1.1, 2.1],
                               [3.1, 4.1]]],
                             dims=('time', 'y', 'x'), coords=dict(y=[0, 1], x=[-2, -1])))))
