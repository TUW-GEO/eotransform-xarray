import pytest

from assertions import assert_raster_allclose
from eotransform_xarray.transformers.raster_add_noise import RasterAddNoise
from factories import make_raster


@pytest.mark.parametrize('method, expected', [
    ('gaussian', [[0.53047171, 0.39600159],
                  [0.57504512, 0.59405647]]),
    ('speckle', [[0.515236, 0.448001],
                 [0.537523, 0.547028]])
])
def test_add_specified_noise_type_to_raster_image(method, expected):
    raster = make_raster([[0.5, 0.5],
                          [0.5, 0.5]])
    noisy = RasterAddNoise(method, seed=42)(raster)
    assert_raster_allclose(noisy, expected)
