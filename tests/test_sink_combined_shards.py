import numpy as np
import pytest
from xarray import DataArray

from eotransform_xarray.transformers import TransformerOfDataArray
from factories import make_raster


class CombineShards(TransformerOfDataArray):
    class MissingPixelCoordinatesError(AttributeError):
        ...

    def __init__(self, canvas: DataArray):
        ...

    def __call__(self, x: DataArray) -> DataArray:
        raise CombineShards.MissingPixelCoordinatesError(f"pixel coordinates missing in attributes:\n{x.attrs}")


def test_combined_shards_raises_error_if_no_pixel_coordinates_are_specified():
    shard = make_raster(make_raster(np.ones((1, 8, 8))))
    with pytest.raises(CombineShards.MissingPixelCoordinatesError):
        CombineShards(canvas=make_raster(np.full((1, 16, 16), np.nan)))(shard)
