from xarray import Dataset, DataArray

from assertions import assert_dataset_identical
from eotransform_xarray.transformers import TransformerOfDataArray
from eotransform_xarray.transformers.apply_to_data_var import ApplyToDataVar
from factories import make_raster


class AddOne(TransformerOfDataArray):
    def __call__(self, x: DataArray) -> DataArray:
        return x + 1


def test_apply_a_transformation_to_the_specified_data_var():
    dataset = Dataset(dict(
        var_0=make_raster([[10]]),
        var_1=make_raster([[20]]),
    ))
    apply_to_data_var = ApplyToDataVar("var_1", AddOne())
    assert_dataset_identical(apply_to_data_var(dataset), Dataset(dict(
        var_0=make_raster([[10]]),
        var_1=make_raster([[21]]),
    )))
