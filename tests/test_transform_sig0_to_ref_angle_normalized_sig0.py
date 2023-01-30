import numpy as np

from assertions import assert_raster_eq
from eotransform_xarray.geometry.degrees import Degree
from eotransform_xarray.transformers.normalize_sig0_to_ref_lia_by_slope import NormalizeSig0ToRefLiaBySlope
from helpers.factories import make_raster


def test_sig0_is_interpolated_to_reference_angle_based_on_slope():
    normalize = NormalizeSig0ToRefLiaBySlope(make_raster([[-0.1]]), make_raster([[20]]), Degree(40))
    assert_raster_eq(normalize(make_raster([[-10]])), make_raster([[-12]]))


def test_mask_all_values_where_slope_plia_and_sig0_are_nan():
    normalize = NormalizeSig0ToRefLiaBySlope(
        make_raster([[np.nan, -0.1],
                     [-0.1, -0.1]]),
        make_raster([[20, np.nan],
                     [20, 20]]), Degree(40))
    assert_raster_eq(normalize(
        make_raster([[-10, -10],
                     [np.nan, -10]])),
        make_raster([[np.nan, np.nan],
                     [np.nan, -12]]))
