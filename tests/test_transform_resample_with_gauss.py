from typing import Tuple

import numpy as np
from affine import Affine
from approval_utilities.utilities.exceptions.exception_collector import gather_all_exceptions_and_throw
from approvaltests.namer import NamerFactory
from pytest_approvaltests_geo import GeoOptions
from xarray import DataArray

from eotransform_xarray.transformers.resample_with_gauss import Swath, Extent, Area, ResampleWithGauss


def test_resample_raster_using_gauss_interpolation(verify_raster_as_geo_tif):
    lons, lats = np.meshgrid([12.0, 16.0], [47.9, 45.2])
    lons = lons.reshape(1, -1)
    lats = lats.reshape(1, -1)
    extent = Extent(4800000, 1200000, 5400000, 1800000)
    transform = Affine.from_gdal(4800000, 3000, 0, 1800000, 0, 3000)
    projection = "+proj=aeqd +lat_0=53 +lon_0=24 +x_0=5837287.81977 +y_0=2121415.69617 +datum=WGS84 +units=m +no_defs"
    in_data = DataArray([[[1, 2, 4, 8]], [[1, 2, 4, np.nan]]], dims=['time', 'parameter', 'value'], coords=(dict(
        time=np.arange(0, 2),
        lon=(('time', 'parameter', 'value'), np.stack([lons, lons])),
        lat=(('time', 'parameter', 'value'), np.stack([lats, lats])))))
    resample = ResampleWithGauss(Swath(lons, lats),
                                 Area("test_area", projection, 200, 200, extent, transform),
                                 sigma=2e5, neighbours=4, lookup_radius=1e6)
    resampled = resample(in_data)
    gather_all_exceptions_and_throw([0, 1], lambda t: verify_raster_as_geo_tif(
        mask_and_scale(resampled[t]),
        options=GeoOptions.from_options(NamerFactory.with_parameters(t))
    ))


def mask_and_scale(a: DataArray) -> DataArray:
    scale_factor = 1e-3
    a /= scale_factor
    a.attrs['scale_factor'] = scale_factor
    a = a.fillna(-9999)
    a.rio.write_nodata(-9999, inplace=True)
    return a.astype(np.int16)
