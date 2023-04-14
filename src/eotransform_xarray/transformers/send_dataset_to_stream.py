from abc import ABC, abstractmethod
from typing import Callable, Any, Optional

from eotransform_xarray.transformers import TransformerOfDataset
from xarray import Dataset, DataArray

ProcessFn = Callable[[DataArray], Any]


class StreamIn(ABC):
    @abstractmethod
    def send(self, *args):
        ...


def identity(a):
    return a


class SendDatasetToStream(TransformerOfDataset):
    def __init__(self, stream: StreamIn, *data_vars_to_send, preprocess: Optional[ProcessFn] = None):
        self._stream = stream
        self._data_vars_to_send = data_vars_to_send
        self._preprocess = preprocess or identity

    def __call__(self, x: Dataset) -> Dataset:
        self._stream.send(*tuple(self._preprocess(x[v]) for v in self._data_vars_to_send))
        return x
