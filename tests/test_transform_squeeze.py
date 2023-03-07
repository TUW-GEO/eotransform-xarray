from xarray import Dataset

from assertions import assert_dataset_identical
from eotransform_xarray.transformers.squeeze import Squeeze
from factories import make_raster


def test_squeeze_dataset_dim():
    dataset = Dataset(dict(
        var_0=make_raster([[[10]]]),
    ))
    squeeze = Squeeze(dim='band', drop=True)
    assert_dataset_identical(squeeze(dataset), Dataset(dict(
        var_0=make_raster([[[10]]]).squeeze('band', drop=True),
    )))


def test_squeeze_dataset_axis():
    dataset = Dataset(dict(
        var_0=make_raster([[[10]]]),
    ))
    squeeze = Squeeze(axis=0, drop=True)
    assert_dataset_identical(squeeze(dataset), Dataset(dict(
        var_0=make_raster([[[10]]]).squeeze(axis=0, drop=True),
    )))
