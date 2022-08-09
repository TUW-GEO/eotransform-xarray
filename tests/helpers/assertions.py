import xarray as xr

from factories import make_raster


def assert_raster_eq(actual, expected):
    xr.testing.assert_equal(actual, make_raster(expected))
