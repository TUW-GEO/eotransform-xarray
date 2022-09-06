from enum import Enum
from typing import Optional

from xarray import DataArray

from eotransform_xarray.sinks import DataArraySink

SHARD_ATTRS_KEY = "shard_attrs"


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
