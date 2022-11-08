import shutil
from datetime import datetime
from typing import Any, Sequence, Optional

import numpy as np
from affine import Affine
from approval_utilities.utilities.exceptions.exception_collector import gather_all_exceptions_and_throw
from approvaltests.namer import NamerFactory
from numpy.typing import ArrayLike
from pytest_approvaltests_geo import GeoOptions
from xarray import DataArray

from eotransform_xarray.storage.storage_using_zarr import StorageUsingZarr
from eotransform_xarray.transformers.resample_with_gauss import Swath, Extent, Area, ResampleWithGauss, \
    ProjectionParameter
from helpers.assertions import assert_data_array_eq

DEFAULT_TEST_EXTENT = Extent(4800000, 1200000, 5400000, 1800000)
DEFAULT_TEST_TRANSFORM = Affine.from_gdal(4800000, 3000, 0, 1800000, 0, 3000)
DEFAULT_TEST_PROJECTION = "+proj=aeqd +lat_0=53 +lon_0=24 +x_0=5837287.81977 +y_0=2121415.69617 +datum=WGS84 +units=m +no_defs"


def test_resample_raster_using_gauss_interpolation(verify_raster_as_geo_tif):
    swath = make_swath([12.0, 16.0], [47.9, 45.2])
    in_data = make_swath_data_array([[[1, 2, 4, 8]], [[1, 2, 4, np.nan]]], swath)

    resample = ResampleWithGauss(swath, make_target_area(200, 200), sigma=2e5, neighbours=4, lookup_radius=1e6,
                                 raster_chunk_sizes=(200, 200))
    resampled = resample(in_data)

    gather_all_exceptions_and_throw([0, 1], lambda t: verify_raster_as_geo_tif(
        mask_and_scale(resampled[t]),
        options=GeoOptions.from_options(NamerFactory.with_parameters(t))
    ))


def make_swath(lons: ArrayLike, lats: ArrayLike) -> Swath:
    lons, lats = np.meshgrid(lons, lats)
    lons = lons.reshape(1, -1)
    lats = lats.reshape(1, -1)
    return Swath(lons, lats)


def make_target_area(columns: int, rows: int) -> Area:
    return Area("test_area", DEFAULT_TEST_PROJECTION, columns, rows, DEFAULT_TEST_EXTENT, DEFAULT_TEST_TRANSFORM)


def make_swath_data_array(values: Any, swath: Swath, ts: Optional[Sequence[datetime]] = None,
                          parameters: Optional[Sequence[str]] = None) -> DataArray:
    values = np.array(values)
    coords = dict(time=ts or np.arange(0, values.shape[0]),
                  lon=(('time', 'parameter', 'value'), np.tile(swath.lons, (*values.shape[:-1], 1))),
                  lat=(('time', 'parameter', 'value'), np.tile(swath.lats, (*values.shape[:-1], 1))))
    if parameters:
        coords['parameter'] = parameters
    return DataArray(values, dims=['time', 'parameter', 'value'], coords=coords)


def mask_and_scale(a: DataArray) -> DataArray:
    scale_factor = 1e-3
    a /= scale_factor
    a.attrs['scale_factor'] = scale_factor
    a = a.fillna(-9999)
    a.rio.write_nodata(-9999, inplace=True)
    return a.astype(np.int16)


def test_resample_raster_with_gauss_uses_max_lookup_radius():
    swath = make_swath([12.0, 16.0], [47.9, 45.2])
    swath.lons[0, -1] = 21.5
    swath.lats[0, -1] = 40.5
    in_data = make_swath_data_array([[[1, 2, 4, 8], [1, 2, 4, np.nan]]], swath)

    resample = ResampleWithGauss(swath, make_target_area(200, 200), sigma=2e5, neighbours=4, lookup_radius=5e5,
                                 raster_chunk_sizes=(200, 200))
    resampled = resample(in_data)

    assert_data_array_eq(resampled[0, 0], resampled[0, 1])


def test_store_resampling_transformation(tmp_path):
    swath = make_swath([12.0, 16.0], [47.9, 45.2])
    in_data = make_swath_data_array([[[1, 2, 4, 8]], [[1, 2, 4, np.nan]]], swath)
    zarr_storage = StorageUsingZarr(tmp_path / "resampling")

    ResampleWithGauss(swath, make_target_area(200, 200), sigma=2e5, neighbours=4, lookup_radius=1e6,
                      raster_chunk_sizes=(100, 100), resampling_parameter_storage=zarr_storage)

    flip_stored_valid_input_bit_at(-1, zarr_storage)
    resample_stored = ResampleWithGauss(swath, make_target_area(200, 200), sigma=2e5, neighbours=4, lookup_radius=1e6,
                                        raster_chunk_sizes=(100, 100), resampling_parameter_storage=zarr_storage)

    resampled_with_last_input_masked = resample_stored(in_data)
    assert_array_eq(resampled_with_last_input_masked[0, 0].values, resampled_with_last_input_masked[1, 0].values)


def assert_array_eq(actual: np.ndarray, expected: np.ndarray):
    np.testing.assert_array_equal(actual, expected)


def flip_stored_valid_input_bit_at(index, zarr_storage):
    projection_params = ProjectionParameter.from_storage(zarr_storage)
    projection_params.in_resampling.load()
    projection_params.out_resampling.load()
    projection_params.in_resampling['mask'][0, index] = not projection_params.in_resampling['mask'][0, index]
    for sub in zarr_storage.path.iterdir():
        shutil.rmtree(sub)
    projection_params.store(zarr_storage)


def test_resample_raster_preserves_coordinates():
    swath = make_swath([12.0, 16.0], [47.9, 45.2])
    in_data = make_swath_data_array([[[1, 2, 4, 8]], [[1, 2, 4, np.nan]]], swath,
                                    ts=[datetime(2022, 10, 20), datetime(2022, 10, 21)],
                                    parameters=['value_name'])

    resample = ResampleWithGauss(swath, make_target_area(200, 200), sigma=2e5, neighbours=4, lookup_radius=5e5,
                                 raster_chunk_sizes=(200, 200))
    resampled = resample(in_data)

    assert_ts_eq(resampled['time'], [datetime(2022, 10, 20), datetime(2022, 10, 21)])
    assert resampled['parameter'][0] == 'value_name'


def assert_ts_eq(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual, dtype='datetime64'), np.array(expected, dtype='datetime64'))


def test_resample_raster_preserves_attributes():
    swath = make_swath([12.0, 16.0], [47.9, 45.2])
    in_data = make_swath_data_array([[[1, 2, 4, 8]], [[1, 2, 4, np.nan]]], swath)
    in_data.attrs = dict(some='attribute')

    resample = ResampleWithGauss(swath, make_target_area(200, 200), sigma=2e5, neighbours=4, lookup_radius=1e6,
                                 raster_chunk_sizes=(200, 200))
    resampled = resample(in_data)

    assert resampled.attrs == dict(some='attribute')
