from abc import ABC

from eotransform.protocol.transformer import Transformer
from xarray import DataArray


class TransformerOfDataArray(Transformer[DataArray, DataArray], ABC):
    ...
