from xarray import DataArray

from eotransform_xarray.geometry.degrees import Degree
from eotransform_xarray.transformers import TransformerOfDataArray


class NormalizeSig0ToRefLiaBySlope(TransformerOfDataArray):
    def __init__(self, slope: DataArray, lia: DataArray, reference_lia: Degree):
        self._slope = slope
        self._lia = lia
        self._reference_lia = reference_lia

    def __call__(self, x: DataArray) -> DataArray:
        return x - self._slope * (self._lia - self._reference_lia.value)
