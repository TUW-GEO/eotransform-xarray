from abc import abstractmethod

from eotransform.protocol.sink import Sink
from xarray import DataArray


class DataArraySink(Sink[DataArray]):
    @abstractmethod
    def __call__(self, x: DataArray) -> None:
        ...
