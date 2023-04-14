from typing import Callable, Any, Optional, Tuple

from eotransform.protocol.stream import StreamIn
from eotransform.transformers.send_to_stream import SendToStream
from xarray import Dataset, DataArray


ProcessFn = Callable[[DataArray], Any]
DataVars = Tuple[Any, ...]


def identity(a):
    return a


class SendDataVarsToStream(SendToStream[DataVars]):
    def __init__(self, stream: StreamIn[DataVars], *data_vars_to_send,
                 preprocess: Optional[ProcessFn] = None):
        super().__init__(stream)
        self._data_vars_to_send = data_vars_to_send
        self._preprocess = preprocess or identity

    def __call__(self, x: Dataset) -> Dataset:
        super().__call__(tuple(self._preprocess(x[v]) for v in self._data_vars_to_send))
        return x
