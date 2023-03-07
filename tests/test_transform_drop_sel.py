from xarray import Dataset

from assertions import assert_dataset_identical
from eotransform_xarray.transformers.drop_sel import DropSel
from factories import make_raster


def test_drop_the_specified_dims():
    dataset = Dataset(dict(
        var_0=make_raster([[[10]]]),
        var_1=make_raster([[[20]], [[30]]])
    ))
    drop_sel = DropSel(dict(band=[2]))
    assert_dataset_identical(drop_sel(dataset), Dataset(dict(
        var_0=make_raster([[[10]]]),
        var_1=make_raster([[[20]]]),
    )))
