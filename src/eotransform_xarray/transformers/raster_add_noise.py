from skimage.util import random_noise
from xarray import DataArray

from eotransform_xarray.transformers import TransformerOfDataArray


class RasterAddNoise(TransformerOfDataArray):
    def __init__(self, method, seed=None):
        self._method = method
        self._seed = seed

    def __call__(self, x: DataArray) -> DataArray:
        noised = random_noise(x.as_numpy(), self._method, seed=self._seed)
        x.data = noised
        return x
