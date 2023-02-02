import numpy as np
import pytest

from assertions import assert_raster_eq
from eotransform_xarray.constants import SOURCE_KEY
from eotransform_xarray.geometry.degrees import Degree
from eotransform_xarray.transformers.normalize_sig0_to_ref_lia_by_slope import NormalizeSig0ToRefLiaBySlope, Engine, \
    ORBIT_KEY, METADATA_KEY, SLOPE_SRC_KEY, LIA_SRC_KEY, SIG0_SRC_KEY, REF_ANGLE_KEY, ENGINE_USED_KEY
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
    normalize = NormalizeSig0ToRefLiaBySlope(make_raster([[-0.1]]), {'095': make_raster([[20]])}, Degree(40))
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


@pytest.mark.parametrize('engine', [Engine.DASK, Engine.NUMBA])
def test_write_normalization_meta_data_in_resulting_array(engine):
    normalize = NormalizeSig0ToRefLiaBySlope(make_raster([[-0.1]], encoding={SOURCE_KEY: 'slope/file.tif'}),
                                             {'095': make_raster([[20]], encoding={SOURCE_KEY: 'plia/file.tif'})},
                                             Degree(40), engine)
    assert normalize(make_raster([[-10]], attrs={ORBIT_KEY: '095'}, encoding={SOURCE_KEY: 'sig0/file.tif'})).attrs == {
        ORBIT_KEY: '095',
        METADATA_KEY: {
            SLOPE_SRC_KEY: 'slope/file.tif',
            LIA_SRC_KEY: 'plia/file.tif',
            SIG0_SRC_KEY: 'sig0/file.tif',
            REF_ANGLE_KEY: 40,
            ENGINE_USED_KEY: engine.name
        }
    }
