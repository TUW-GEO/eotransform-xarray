import shutil
from datetime import datetime
from typing import Optional

import numpy as np
import pytest
import rioxarray
from approval_utilities.utilities.exceptions.exception_collector import gather_all_exceptions_and_throw
from approvaltests import Options
from approvaltests.namer import NamerFactory
from pytest_approvaltests_geo import GeoOptions
from xarray import DataArray

from eotransform_xarray.geometry.pixel_transforms import pixel_centers_for_geotransform_extent
from eotransform_xarray.storage.storage_using_zarr import StorageUsingZarr
from eotransform_xarray.transformers.resample_with_gauss import ResampleWithGauss, \
    ProjectionParameter, ProcessingConfig, DaskConfig, NumbaConfig
from factories import make_swath, make_target_area, make_swath_data_array
from helpers.assertions import assert_data_array_eq


@pytest.fixture(params=[NumbaConfig(), DaskConfig((200, 200))])
def processing_config(request):
    engine = request.param
    loading = dict(scheduler='single-threaded') if engine == 'numba' else None
    return ProcessingConfig(resampling_engine=engine,
                            num_lookup_segments=2,
                            load_in_resampling_params=loading,
                            load_out_resampling_params=loading)


@pytest.fixture
def engine_type(processing_config):
    return processing_config.resampling_engine.type


@pytest.fixture
def verify_raster(verify_geo_tif_with_namer, tmp_path_factory):
    def _verify_fn(tile: DataArray,
                   *,  # enforce keyword arguments - https://www.python.org/dev/peps/pep-3102/
                   options: Optional[Options] = None):
        tile_file = tmp_path_factory.mktemp("raster_as_geo_tif") / "raster.tif"
        tile.rio.to_raster(tile_file)
        verify_geo_tif_with_namer(tile_file, options.namer, options=GeoOptions.from_options(options))

    return _verify_fn


def test_resample_raster_using_gauss_interpolation(verify_raster, processing_config, engine_type):
    swath = make_swath([12.0, 16.0], [47.9, 45.2])
    in_data = make_swath_data_array([[[1, 2, 4, 8]], [[1, 2, 4, np.nan]]], swath)

    resample = ResampleWithGauss(swath, make_target_area(200, 200), sigma=2e5, neighbours=4, lookup_radius=1e6,
                                 processing_config=processing_config)
    resampled = resample(in_data)

    gather_all_exceptions_and_throw([0, 1], lambda t: verify_raster(
        mask_and_scale(resampled[t]),
        options=NamerFactory.with_parameters(t, engine_type).for_file.with_extension('.tif')
    ))


def mask_and_scale(a: DataArray) -> DataArray:
    scale_factor = 1e-3
    a /= scale_factor
    a.attrs['scale_factor'] = scale_factor
    a = a.fillna(-9999)
    a.rio.write_nodata(-9999, inplace=True)
    return a.astype(np.int16)


def test_resample_raster_with_gauss_uses_max_lookup_radius(processing_config):
    swath = make_swath([12.0, 16.0], [47.9, 45.2])
    swath.lons[0, -1] = 21.5
    swath.lats[0, -1] = 40.5
    in_data = make_swath_data_array([[[1, 2, 4, 8], [1, 2, 4, np.nan]]], swath)

    resample = ResampleWithGauss(swath, make_target_area(200, 200), sigma=2e5, neighbours=4, lookup_radius=5e5,
                                 processing_config=processing_config)
    resampled = resample(in_data)

    assert_data_array_eq(resampled[0, 0], resampled[0, 1])


def test_store_resampling_transformation(tmp_path, processing_config):
    swath = make_swath([12.0, 16.0], [47.9, 45.2])
    in_data = make_swath_data_array([[[1, 2, 4, 8]], [[1, 2, 4, np.nan]]], swath)
    zarr_storage = StorageUsingZarr(tmp_path / "resampling")
    processing_config.parameter_storage = zarr_storage

    ResampleWithGauss(swath, make_target_area(200, 200), sigma=2e5, neighbours=4, lookup_radius=1e6,
                      processing_config=processing_config)

    flip_stored_valid_input_bit_at(-1, zarr_storage)
    resample_stored = ResampleWithGauss(swath, make_target_area(200, 200), sigma=2e5, neighbours=4, lookup_radius=1e6,
                                        processing_config=processing_config)

    resampled_with_last_input_masked = resample_stored(in_data)
    assert_array_eq(resampled_with_last_input_masked[0, 0].values, resampled_with_last_input_masked[1, 0].values)


def assert_array_eq(actual: np.ndarray, expected: np.ndarray):
    np.testing.assert_array_equal(actual, expected)


def flip_stored_valid_input_bit_at(index, zarr_storage):
    projection_params = ProjectionParameter.from_storage(zarr_storage)
    projection_params.in_resampling.load()
    projection_params.out_resampling.load()
    projection_params.in_resampling['mask'][index, 0] = not projection_params.in_resampling['mask'][index, 0]
    for sub in zarr_storage.path.iterdir():
        shutil.rmtree(sub)
    projection_params.store(zarr_storage)


def test_resample_raster_preserves_coordinates(processing_config):
    swath = make_swath([12.0, 16.0], [47.9, 45.2])
    in_data = make_swath_data_array([[[1, 2, 4, 8]], [[1, 2, 4, np.nan]]], swath,
                                    ts=[datetime(2022, 10, 20), datetime(2022, 10, 21)],
                                    parameters=['value_name'])

    resample = ResampleWithGauss(swath, make_target_area(200, 200), sigma=2e5, neighbours=4, lookup_radius=5e5,
                                 processing_config=processing_config)
    resampled = resample(in_data)

    assert_ts_eq(resampled['time'], [datetime(2022, 10, 20), datetime(2022, 10, 21)])
    assert resampled['parameter'][0] == 'value_name'


def assert_ts_eq(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual, dtype='datetime64'), np.array(expected, dtype='datetime64'))


def test_resampled_raster_assigns_correct_spatial_coordinates(verify_raster, processing_config, engine_type, tmp_path):
    swath = make_swath([12.0, 16.0], [47.9, 45.2])
    in_data = make_swath_data_array([[[1, 2, 4, 8]], [[1, 2, 4, np.nan]]], swath)

    area = make_target_area(200, 200)
    resample = ResampleWithGauss(swath, area, sigma=2e5, neighbours=4, lookup_radius=1e6,
                                 processing_config=processing_config)
    resampled = resample(in_data)
    resampled[0, 0].rio.to_raster(tmp_path / "resampled.tif")
    encoded_decoded = rioxarray.open_rasterio(tmp_path / "resampled.tif", parse_coordinates=True)

    y, x = pixel_centers_for_geotransform_extent(area.transform, 200, 200)
    assert_array_eq(resampled.y, y)
    assert_array_eq(resampled.x, x)
    assert_spatial_coords_eq(resampled, encoded_decoded)


def assert_spatial_coords_eq(actual, expected):
    assert_array_eq(actual.y, expected.y)
    assert_array_eq(actual.x, expected.x)

def test_resample_raster_preserves_attributes(processing_config):
    swath = make_swath([12.0, 16.0], [47.9, 45.2])
    in_data = make_swath_data_array([[[1, 2, 4, 8]], [[1, 2, 4, np.nan]]], swath)
    in_data.attrs = dict(some='attribute')

    resample = ResampleWithGauss(swath, make_target_area(200, 200), sigma=2e5, neighbours=4, lookup_radius=1e6,
                                 processing_config=processing_config)
    resampled = resample(in_data)

    assert resampled.attrs == dict(some='attribute')
