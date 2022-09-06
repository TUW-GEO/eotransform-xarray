from enum import Enum
from typing import Optional

import rioxarray  # noqa # pylint: disable=unused-import
from xarray import DataArray

from eotransform_xarray.sinks import DataArraySink

SHARD_ATTRS_KEY = "shard_attrs"


class CombineShards(DataArraySink):
    """
    Combine raster shards into one big canvas by assigning them to the canvas based on their exact coordinates.
    Different methods of assignment are supported.

    >>> combined = CombineShards(DataArray([[0, 0, 0],
    ...                                     [0, 0, 0],
    ...                                     [0, 0, 0]], coords=dict(y=[0, 1, 2], x=[0, 1, 2])))
    >>> combined(DataArray([[1, 1],
    ...                     [1, 1]], coords=dict(y=[0, 1], x=[0, 1]), attrs=dict(foo=1)))
    >>> combined(DataArray([[2, 2],
    ...                     [2, 2]], coords=dict(y=[1, 2], x=[1, 2]), attrs=dict(foo=2)))
    >>> combined.canvas
    <xarray.DataArray (y: 3, x: 3)>
    array([[1, 1, 0],
           [1, 2, 2],
           [0, 2, 2]])
    Coordinates:
      * y        (y) int64 0 1 2
      * x        (x) int64 0 1 2
    Attributes:
        shard_attrs:  [{'foo': 1}, {'foo': 2}]
    """
    class Method(Enum):
        ASSIGN = 'assign'
        OR = 'or'

    class CoordinateSystemMismatchError(AttributeError):
        ...

    def __init__(self, canvas: DataArray, method: Optional[Method] = Method.ASSIGN):
        """
        :param canvas: the raster to assign the shards to (needs to be in the same coordinate system)
        :param method: the method used to apply shards to the canvas (default: assign).
            - Method.ASSIGN: directly assigns the shard data to the canvas overwriting previous ones.
            - Method.OR: or together the canvas and the shard. This requires boolean data.
        """
        self._canvas = canvas
        self._target_crs = canvas.rio.crs
        self._method = method
        self._canvas.attrs[SHARD_ATTRS_KEY] = []

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

        self._canvas.attrs[SHARD_ATTRS_KEY].append(x.attrs)
