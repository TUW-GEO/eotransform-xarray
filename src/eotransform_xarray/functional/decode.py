import numpy as np
from numba import guvectorize, uint8, float64, float32, int16, uint16, int32, uint32, int64, uint64
from xarray import DataArray

SCALE_FACTOR_KEY = "scale_factor"
NO_DATA_KEY = "_FillValue"


@guvectorize([
    (uint8, float64, uint8, float32[:]),
    (int16, float64, int16, float32[:]),
    (uint16, float64, uint16, float32[:]),
    (int32, float64, int32, float64[:]),
    (uint32, float64, uint32, float64[:]),
    (int64, float64, int64, float64[:]),
    (uint64, float64, uint64, float64[:]),
    (float32, float64, float32, float32[:]),
    (float64, float64, float32, float64[:]),
], '(),(),()->()', target='parallel')
def decode(a, scale_factor, no_data, out) -> None:
    out[0] = a * scale_factor if a != no_data else np.nan


def decode_array(x):
    return DataArray(decode(x.values, x.attrs[SCALE_FACTOR_KEY], x.dtype.type(x.attrs['_FillValue'])),
                     x.coords, x.dims, x.name, x.attrs)
