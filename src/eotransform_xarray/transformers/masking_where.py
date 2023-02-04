from typing import Union, Callable, Any

from xarray import DataArray, Dataset

from eotransform_xarray.transformers import TransformerOfXArrayData, XArrayData

MaskingSource = Union[Callable[[Any], Any], DataArray, Dataset]


class MaskWhere(TransformerOfXArrayData):
    def __init__(self, predicate: MaskingSource, replacement_value: Any):
        self._predicate = predicate
        self._replacement_value = replacement_value

    def __call__(self, x: XArrayData) -> XArrayData:
        return x.where(self._predicate, self._replacement_value)
