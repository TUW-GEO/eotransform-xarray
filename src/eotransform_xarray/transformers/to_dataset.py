from typing import Optional

from eotransform.protocol.transformer import Transformer
from xarray import DataArray, Dataset


class ToDataset(Transformer[DataArray, Dataset]):
    class MissingDataError(ValueError):
        ...

    def __init__(self, name: Optional[str] = None, promote_attrs: bool = False):
        self._name = name
        self._promote_attrs = promote_attrs

    def __call__(self, x: DataArray) -> Dataset:
        if not x.name and not self._name:
            msg = f"No name has been defined in the transforms ctor, and the array provided has no name either:\n{x}"
            raise ToDataset.MissingDataError(msg)
        return x.to_dataset(name=self._name, promote_attrs=self._promote_attrs)
