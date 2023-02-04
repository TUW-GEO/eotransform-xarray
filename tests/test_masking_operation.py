import numpy as np
from xarray import Dataset

from assertions import assert_data_array_identical, assert_dataset_identical
from eotransform_xarray.transformers.masking_where import MaskWhere
from factories import make_raster


def test_mask_where_predicate_is_false_with_replacement_value():
    masking = MaskWhere(lambda p: p < 1, 42)
    assert_data_array_identical(masking(make_raster([[0, 1],
                                                     [1, 0]])), make_raster([[0, 42],
                                                                             [42, 0]]))


def test_mask_datasets():
    masking = MaskWhere(lambda p: p < 1, 42)
    assert_dataset_identical(
        masking(Dataset(dict(
            a=make_raster([[0, 1],
                           [1, 0]]),
            b=make_raster([[1, 0],
                           [0, 1]])
        ))), Dataset(dict(
            a=make_raster([[0, 42],
                           [42, 0]]),
            b=make_raster([[42, 0],
                           [0, 42]])
        )))


def test_mask_with_bool_data_array():
    masking = MaskWhere(make_raster([[False, True],
                                     [True, False]]), np.nan)
    assert_data_array_identical(masking(make_raster([[0, 1],
                                                     [1, 0]])), make_raster([[np.nan, 1],
                                                                             [1, np.nan]]))


def test_mask_with_bool_dataset():
    masking = MaskWhere(Dataset(dict(
        a=make_raster([[False, True],
                       [True, False]]),
        b=make_raster([[False, False],
                       [True, True]])
    )), 42)
    assert_dataset_identical(
        masking(Dataset(dict(
            a=make_raster([[0, 1],
                           [1, 0]]),
            b=make_raster([[1, 0],
                           [0, 1]])
        ))), Dataset(dict(
            a=make_raster([[42, 1],
                           [1, 42]]),
            b=make_raster([[42, 42],
                           [0, 1]])
        )))
