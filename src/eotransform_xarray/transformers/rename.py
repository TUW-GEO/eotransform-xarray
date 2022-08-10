from eotransform.protocol.transformer import Transformer
from xarray import DataArray


class Rename(Transformer[DataArray, DataArray]):
    def __init__(self, new_name: str):
        self._new_name = new_name

    def __call__(self, x: DataArray) -> DataArray:
        return x.rename(self._new_name)
