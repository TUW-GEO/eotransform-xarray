from typing import Optional

from eotransform.protocol.transformer import Transformer
from xarray import DataArray, Dataset
from xarray.core.merge import merge_attrs
from xarray.core.types import CombineAttrsOptions


class SelectVariable(Transformer[Dataset, DataArray]):
    def __init__(self, selection: str, combine_dataset_attrs: Optional[CombineAttrsOptions] = None):
        self._selection = selection
        self._combine_attrs = combine_dataset_attrs

    def __call__(self, x: Dataset) -> DataArray:
        a = x[self._selection]
        if not self._combine_attrs:
            return a
        a.attrs = merge_attrs([a.attrs, x.attrs], self._combine_attrs)
        return a
