from dataclasses import dataclass

from xarray import DataArray

from assertions import assert_raster_eq
from eotransform_xarray.transformers import TransformerOfDataArray
from helpers.factories import make_raster


@dataclass
class Degree:
    value: float


class NormalizeSig0ToRefLiaBySlope(TransformerOfDataArray):
    def __init__(self, slope: DataArray, lia: DataArray, reference_lia: Degree):
        self._slope = slope
        self._lia = lia
        self._reference_lia = reference_lia

    def __call__(self, x: DataArray) -> DataArray:
        return x - self._slope * (self._lia - self._reference_lia.value)


def test_sig0_is_interpolated_to_reference_angle_based_on_slope():
    normalize = NormalizeSig0ToRefLiaBySlope(make_raster([[-0.1]]), make_raster([[20]]), Degree(40))
    assert_raster_eq(normalize(make_raster([[-10]])), make_raster([[-12]]))
