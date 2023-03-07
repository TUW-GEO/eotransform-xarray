import xarray as xr
from numpy.typing import NDArray
from xarray import DataArray

from eotransform_xarray.transformers.reduce import Reduce


def test_reduce_data_array_based_on_reducer_function():
    def max_reduction(a: NDArray, axis: int):
        return a.max(axis)

    reducer = Reduce(max_reduction, 'time')
    reduced = reducer(make_ts_array([[[0, 1],
                                      [2, 3]],
                                     [[1, 0],
                                      [3, 2]]]))
    assert_arrays_identical(reduced, make_folded_array([[1, 1],
                                                        [3, 3]]))


def make_ts_array(values, attrs=None):
    return DataArray(values, dims=('time', 'y', 'x'), attrs=attrs)


def make_folded_array(values):
    return DataArray(values, dims=('y', 'x'))


def assert_arrays_identical(actual, expected):
    xr.testing.assert_identical(actual, expected)


def test_replicates_data_array_reduce_interface():
    def add_mul(a: NDArray, axis: int, mul: int):
        return a.sum(axis) * mul

    reducer = Reduce(add_mul, axis=0, keep_attrs=True, keep_dims=True, mul=2)
    reduced = reducer(make_ts_array([[[0, 1],
                                      [2, 3]],
                                     [[1, 0],
                                      [3, 2]]], attrs=dict(some='attr')))
    assert_arrays_identical(reduced, make_ts_array([[[2, 2],
                                                     [10, 10]]], attrs=dict(some='attr')))
