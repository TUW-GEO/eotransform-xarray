import numpy as np
import pytest

from eotransform_xarray.functional.decode import SCALE_FACTOR_KEY, NO_DATA_KEY
from eotransform_xarray.transformers.decoder import Decoder
from factories import make_raster
from test_transform_reduce import assert_arrays_identical


@pytest.mark.parametrize('in_dtype, out_dtype', [
    (np.int8, np.float32),
    (np.uint8, np.float32),
    (np.int16, np.float32),
    (np.uint16, np.float32),
    (np.int32, np.float64),
    (np.uint32, np.float64),
    (np.int64, np.float64),
    (np.uint64, np.float64),
    (np.float32, np.float32),
    (np.float64, np.float64),
])
def test_decode_array_based_on_attributes(in_dtype, out_dtype):
    decoder = Decoder()
    encoding = {SCALE_FACTOR_KEY: 0.5, NO_DATA_KEY: 255}
    assert_arrays_identical(decoder(make_raster(np.array([[100, 255]], dtype=in_dtype), attrs=encoding)),
                            make_raster(np.array([[50.0, np.nan]], dtype=out_dtype), attrs=encoding))
