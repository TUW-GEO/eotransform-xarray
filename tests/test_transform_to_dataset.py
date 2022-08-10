import pytest

from assertions import assert_dataset_identical
from eotransform_xarray.transformers.to_dataset import ToDataset
from factories import make_raster, make_dataset


def test_convert_data_array_to_data_set():
    array = make_raster([[0]], name='raster')
    assert_dataset_identical(ToDataset()(array), make_dataset({'raster': array}))


def test_error_if_trying_to_convert_raster_without_name_and_providing_no_name():
    array = make_raster([[0]], name=None)
    with pytest.raises(ToDataset.MissingDataError):
        ToDataset()(array)


def test_provide_name_via_ctor():
    array = make_raster([[0]], name=None)
    assert_dataset_identical(ToDataset('raster')(array), make_dataset({'raster': array}))


def test_promote_attributes():
    array = make_raster([[0]], name='raster', attrs=dict(some='attribute'))
    converted_ds = ToDataset(promote_attrs=True)(array)
    assert_dataset_identical(converted_ds, make_dataset({'raster': array}, attrs=dict(some='attribute')))
