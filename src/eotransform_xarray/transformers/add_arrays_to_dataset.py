from typing import Iterable

from eotransform.protocol.transformer import Transformer
from xarray import DataArray, Dataset


class AddArraysToDataset(Transformer[Dataset, Dataset]):
    class MissingDataError(ValueError):
        ...

    def __init__(self, arrays: Iterable[DataArray]):
        self._arrays = arrays

    def __call__(self, x: Dataset) -> Dataset:
        for a in self._arrays:
            if not a.name:
                raise AddArraysToDataset.MissingDataError(f"The arrays added must provide names, but this didn't:\n{a}")
            x[a.name] = a
        return x
