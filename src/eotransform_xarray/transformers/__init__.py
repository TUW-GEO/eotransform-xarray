from abc import ABC
from typing import Union

from eotransform.protocol.transformer import Transformer
from xarray import DataArray, Dataset

XArrayData = Union[DataArray, Dataset]


class TransformerOfDataArray(Transformer[DataArray, DataArray], ABC):
    ...


class TransformerOfXArrayData(Transformer[XArrayData, XArrayData], ABC):
    ...
