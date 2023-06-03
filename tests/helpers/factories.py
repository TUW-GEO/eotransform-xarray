from datetime import datetime
from typing import Dict, Optional, Any, Sequence

import numpy as np
import rasterio
from affine import Affine
from eotransform_pandas.filesystem.gather import gather_files
from eotransform_pandas.filesystem.naming.geopathfinder_conventions import yeoda_naming_convention
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename
from numpy._typing import ArrayLike
from xarray import DataArray, Dataset

from eotransform_xarray.transformers.resample_with_gauss import Swath, Area, Extent

DEFAULT_TEST_EXTENT = Extent(4800000, 1200000, 5400000, 1800000)
DEFAULT_TEST_TRANSFORM = Affine.from_gdal(4800000, 3000, 0, 1800000, 0, 3000)
DEFAULT_TEST_PROJECTION = "+proj=aeqd +lat_0=53 +lon_0=24 +x_0=5837287.81977 +y_0=2121415.69617 +datum=WGS84 +units=m +no_defs"


def make_raster(values, name=None, dims=None, coords=None, attrs=None, encoding=None, add_coords=None):
    values = _ensure_band_dimension(values)
    dims = dims or ['band', 'y', 'x']
    coords = coords or dict(band=np.arange(values.shape[0]) + 1,
                            y=np.arange(values.shape[1]),
                            x=np.arange(values.shape[2]),
                            spatial_ref=0)
    add_coords = add_coords or {}
    coords = {**coords, **add_coords}
    array = DataArray(values, dims=dims, coords=coords)
    if attrs:
        array.attrs = attrs
    if name:
        array.name = name
    if encoding:
        array.encoding = encoding
    return array


def _ensure_band_dimension(expected):
    expected = np.array(expected)
    if expected.ndim == 2:
        expected = expected[np.newaxis, ...]
    return expected


def iota_arrays(start, periods, shape):
    for i in range(start, start + periods):
        yield np.full(shape, i)


def generate_yeoda_geo_tiffs(root, date_range, arrays, attrs=None, legacy=False):
    for i, (date, array) in enumerate(zip(date_range, arrays)):
        yeoda_name = YeodaFilename(dict(datetime_1=date))
        da = make_raster(array, name=f"iota_{i}", attrs=attrs)
        if legacy:
            with rasterio.open(root / str(yeoda_name), 'w', 'GTiff', da.shape[1], da.shape[2], da.shape[0], da.rio.crs,
                               da.rio.transform(), da.dtype, da.rio.nodata) as dst:
                dst.write(da.values[0], 1)
                dst.update_tags(**da.attrs)
        else:
            da.rio.to_raster(root / str(yeoda_name))

    return gather_files(root, yeoda_naming_convention, index='datetime_1')


def make_dataset(variables: Dict[str, DataArray], attrs: Optional[Dict] = None) -> Dataset:
    return Dataset(variables, attrs=attrs)


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
