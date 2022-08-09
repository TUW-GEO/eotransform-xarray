import os
from contextlib import contextmanager

import pandas as pd
import psutil
import xarray as xr

from factories import make_raster


def assert_raster_eq(actual, expected):
    xr.testing.assert_equal(actual, make_raster(expected))


def assert_data_array_eq(actual, expected):
    xr.testing.assert_equal(actual, expected)
    assert actual.attrs == expected.attrs


@contextmanager
def assert_memory_ratio(expected_ratio, operation):
    initial_mem = measure_current_mem()
    yield
    following_mem = measure_current_mem()
    ratio = following_mem / initial_mem
    assert operation(ratio, expected_ratio), ratio


def measure_current_mem():
    return psutil.Process(os.getpid()).memory_info().rss


def assert_data_frame_eq(actual, expected):
    pd.testing.assert_frame_equal(actual, expected)
