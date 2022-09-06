from enum import Enum
from typing import Optional

import numpy as np
import pytest
import rioxarray  # noqa # pylint: disable=unused-import
from xarray import DataArray

from assertions import assert_data_array_eq
from eotransform_xarray.sinks import DataArraySink
from factories import make_raster

SHARD_RECT_KEY = "shard_rect"


class CombineShards(DataArraySink):
    class Method(Enum):
        ASSIGN = 'assign'
        OR = 'or'

    class CoordinateSystemMismatchError(AttributeError):
        ...

    def __init__(self, canvas: DataArray, method: Optional[Method] = Method.ASSIGN):
        self._canvas = canvas
        self._target_crs = canvas.rio.crs
        self._method = method

    @property
    def canvas(self) -> DataArray:
        return self._canvas

    def __call__(self, x: DataArray) -> None:
        if x.rio.crs != self._target_crs:
            raise CombineShards.CoordinateSystemMismatchError(
                f"coordinate system of canvas {self._canvas.rio.crs} doesn't match to shard {x.rio.crs}")

        if self._method == CombineShards.Method.OR:
            self._canvas.loc[dict(y=x.y, x=x.x)] |= x
        else:
            self._canvas.loc[dict(y=x.y, x=x.x)] = x


def test_combine_shards_raises_error_if_coordinates_do_not_match():
    shard = make_geo_raster(np.ones((1, 8, 8)), (np.arange(1015270, 1015278), np.arange(4897030, 4897038)), 'EPSG:3857')
    with pytest.raises(CombineShards.CoordinateSystemMismatchError):
        canvas = make_geo_raster(np.full((1, 16, 16), np.nan), (np.linspace(12.6, 12.76, 16),
                                                                np.linspace(42.4, 42.56, 16)), 'EPSG:4326')
        CombineShards(canvas=canvas)(shard)


def make_geo_raster(values, yx_coords, crs, dtype=None):
    values = np.array(values, dtype=dtype)
    coords = dict(band=np.arange(values.shape[0]) + 1,
                  y=yx_coords[0],
                  x=yx_coords[1],
                  spatial_ref=0)
    r = make_raster(values, coords=coords)
    r.rio.write_crs(crs, inplace=True)
    return r


def test_combine_shards_write_pixels_to_matching_coordinates():
    canvas_coords = dict(yx_coords=(np.arange(1015270, 1015274), np.arange(4897030, 4897034)), crs='EPSG:3857')
    combined = CombineShards(canvas=make_geo_raster(np.zeros((1, 4, 4)), **canvas_coords))

    shard_tl = make_geo_raster(np.ones((1, 3, 2)), (np.arange(1015270, 1015273),
                                                    np.arange(4897030, 4897032)), 'EPSG:3857')
    combined(shard_tl)
    assert_data_array_eq(combined.canvas, make_geo_raster([[[1, 1, 0, 0],
                                                            [1, 1, 0, 0],
                                                            [1, 1, 0, 0],
                                                            [0, 0, 0, 0]]], **canvas_coords))

    shard_br = make_geo_raster(np.ones((1, 3, 2)), (np.arange(1015271, 1015274),
                                                    np.arange(4897032, 4897034)), 'EPSG:3857')
    combined(shard_br)
    assert_data_array_eq(combined.canvas, make_geo_raster([[[1, 1, 0, 0],
                                                            [1, 1, 1, 1],
                                                            [1, 1, 1, 1],
                                                            [0, 0, 1, 1]]], **canvas_coords))


def test_combine_shards_using_or_assignment_method():
    canvas_coords = dict(yx_coords=(np.arange(1015270, 1015274), np.arange(4897030, 4897034)), crs='EPSG:3857')
    combined = CombineShards(make_geo_raster(np.zeros((1, 4, 4)), **canvas_coords, dtype=np.bool_), CombineShards.Method.OR)

    shard_a = make_geo_raster([[[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]]], (np.arange(1015270, 1015273),
                                              np.arange(4897030, 4897033)), 'EPSG:3857', dtype=np.bool_)
    shard_b = make_geo_raster([[[0, 0, 0],
                                [0, 1, 1],
                                [0, 1, 1]]], (np.arange(1015271, 1015274),
                                              np.arange(4897031, 4897034)), 'EPSG:3857', dtype=np.bool_)
    combined(shard_a)
    combined(shard_b)
    assert_data_array_eq(combined.canvas, make_geo_raster([[[1, 1, 1, 0],
                                                            [1, 1, 1, 0],
                                                            [1, 1, 1, 1],
                                                            [0, 0, 1, 1]]], **canvas_coords, dtype=np.bool_))
