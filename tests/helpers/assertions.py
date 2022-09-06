import os
from contextlib import contextmanager

import psutil
import xarray as xr

from factories import make_raster


def assert_raster_eq(actual, expected):
    xr.testing.assert_equal(actual, make_raster(expected))


def assert_raster_allclose(actual, expected):
    xr.testing.assert_allclose(actual, make_raster(expected))


def assert_data_array_eq(actual, expected):
    xr.testing.assert_equal(actual, expected)


def assert_data_array_identical(actual, expected):
    xr.testing.assert_identical(actual, expected)


@contextmanager
def assert_memory_ratio(expected_ratio, operation):
    initial_mem = measure_current_mem()
    yield
    following_mem = measure_current_mem()
    ratio = following_mem / initial_mem
    assert operation(ratio, expected_ratio), ratio


def measure_current_mem():
    return psutil.Process(os.getpid()).memory_info().rss


def assert_dataset_identical(actual, expected):
    xr.testing.assert_identical(actual, expected)
