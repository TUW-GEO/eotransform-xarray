from typing import Dict, Optional

import numpy as np
import rasterio
from eotransform_pandas.filesystem.gather import gather_files
from eotransform_pandas.filesystem.naming.geopathfinder_conventions import yeoda_naming_convention
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename
from xarray import DataArray, Dataset


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
