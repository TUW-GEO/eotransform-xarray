from eotransform_xarray.transformers.rename import Rename
from factories import make_raster


def test_rename_data_array():
    assert Rename('Herbert')(make_raster([[0]])).name == "Herbert"
