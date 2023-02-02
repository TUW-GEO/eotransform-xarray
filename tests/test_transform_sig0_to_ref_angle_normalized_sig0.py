import numpy as np
import pytest

from assertions import assert_raster_eq
from eotransform_xarray.geometry.degrees import Degree
from eotransform_xarray.transformers.normalize_sig0_to_ref_lia_by_slope import NormalizeSig0ToRefLiaBySlope, Engine, \
    ORBIT_KEY
from helpers.factories import make_raster


@pytest.mark.parametrize('engine', [Engine.DASK, Engine.NUMBA])
def test_sig0_is_interpolated_to_reference_angle_based_on_slope(engine):
    normalize = NormalizeSig0ToRefLiaBySlope(make_raster([[-0.1]]), {
        '095': make_raster([[20]]),
        '015': make_raster([[10]]),
    }, Degree(40), engine)
    assert_raster_eq(normalize(make_raster([[-10]], attrs={ORBIT_KEY: '095'})), make_raster([[-12]]))
    assert_raster_eq(normalize(make_raster([[-10]], attrs={ORBIT_KEY: '015'})), make_raster([[-13]]))


def test_raise_an_error_if_no_lai_is_found_for_orbit():
    normalize = NormalizeSig0ToRefLiaBySlope(make_raster([[-0.1]]), {}, Degree(40))
    with pytest.raises(NormalizeSig0ToRefLiaBySlope.MissingLiaError):
        normalize(make_raster([[-10]], attrs={ORBIT_KEY: '095'}))


def test_raise_an_error_if_input_does_not_define_an_orbit():
    normalize = NormalizeSig0ToRefLiaBySlope(make_raster([[-0.1]]), {
        '095': make_raster([[20]]),
    }, Degree(40))
    with pytest.raises(NormalizeSig0ToRefLiaBySlope.MissingOrbitInfoError):
        normalize(make_raster([[-10]]))

@pytest.mark.parametrize('engine', [Engine.DASK, Engine.NUMBA])
def test_mask_all_values_where_slope_plia_and_sig0_are_nan(engine):
    normalize = NormalizeSig0ToRefLiaBySlope(
        make_raster([[np.nan, -0.1],
                     [-0.1, -0.1]]),
        {'095': make_raster([[20, np.nan],
                             [20, 20]])}, Degree(40), engine)
    assert_raster_eq(normalize(
        make_raster([[-10, -10],
                     [np.nan, -10]], attrs={ORBIT_KEY: '095'})),
        make_raster([[np.nan, np.nan],
                     [np.nan, -12]]))
