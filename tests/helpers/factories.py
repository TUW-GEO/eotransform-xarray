import numpy as np

from xarray import DataArray


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
