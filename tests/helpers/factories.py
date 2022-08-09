import numpy as np
import pandas as pd
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename
from xarray import DataArray

from eotransform_xarray.filesystem.gather import yeoda_naming_convention, gather_files


def make_raster(values, name=None, dims=None, coords=None, attrs=None):
    values = _ensure_band_dimension(values)
    dims = dims or ['band', 'y', 'x']
    coords = coords or dict(band=np.arange(values.shape[0]) + 1,
                            y=np.arange(values.shape[1]),
                            x=np.arange(values.shape[2]),
                            spatial_ref=0)
    array = DataArray(values, dims=dims, coords=coords)
    if attrs:
        array.attrs = attrs
    if name:
        array.name = name
    return array


def _ensure_band_dimension(expected):
    expected = np.array(expected)
    if expected.ndim == 2:
        expected = expected[np.newaxis, ...]
    return expected


def iota_arrays(start, periods, shape):
    for i in range(start, start + periods):
        yield np.full(shape, i)


def generate_yeoda_geo_tiffs(root, date_range, arrays):
    for i, (date, array) in enumerate(zip(date_range, arrays)):
        yeoda_name = YeodaFilename(dict(datetime_1=date))
        da = make_raster(array, name=f"iota_{i}", attrs=dict(light_direction=[1, 1, 1]))
        da.rio.to_raster(root / str(yeoda_name))

    return gather_files(root, yeoda_naming_convention, index='datetime_1')


def make_files_dataset(index, **columns):
    df = pd.DataFrame(data=columns)
    df.set_index(index, inplace=True)
    return df
