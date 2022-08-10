import pytest

from assertions import assert_dataset_identical
from eotransform_xarray.transformers.add_arrays_to_dataset import AddArraysToDataset
from factories import make_raster, make_dataset


def test_add_arrays_to_dataset():
    dataset = make_raster([[0]], name='var_0').to_dataset()
    array_1 = make_raster([[1]], name='var_1')
    array_2 = make_raster([[2]], name='var_2')

    dataset = AddArraysToDataset([array_1, array_2])(dataset)
    assert_dataset_identical(dataset, make_dataset(dict(var_0=make_raster([[0]]),
                                                        var_1=make_raster([[1]]),
                                                        var_2=make_raster([[2]]))))


def test_error_when_arrays_do_not_provide_names():
    dataset = make_raster([[0]], name='var_0').to_dataset()
    array_1 = make_raster([[1]], name=None)
    with pytest.raises(AddArraysToDataset.MissingDataError):
        AddArraysToDataset([array_1])(dataset)
