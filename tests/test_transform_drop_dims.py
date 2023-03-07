from xarray import Dataset

from assertions import assert_dataset_identical
from eotransform_xarray.transformers.drop_dims import DropDims
from factories import make_raster


def test_drop_the_specified_dims():
    dataset = Dataset(dict(
        var_0=make_raster([[[10]]]).expand_dims(time=[42]),
        var_1=make_raster([[[20]]])
    ))
    drop_dims = DropDims("time")
    assert_dataset_identical(drop_dims(dataset), Dataset(dict(
        var_1=make_raster([[[20]]]),
    )))