from typing import Union, Callable, Any

import numpy as np
from xarray import DataArray, Dataset

from eotransform_xarray.transformers import TransformerOfXArrayData, XArrayData

MaskingSource = Union[Callable[[Any], Any], DataArray, Dataset]


class MaskWhere(TransformerOfXArrayData):
    def __init__(self, predicate: MaskingSource, replacement_value: Any, invert=False):
        self._predicate = predicate
        self._replacement_value = replacement_value
        self._invert = invert

    def __call__(self, x: XArrayData) -> XArrayData:
        if self._invert:
            return x.where(_not(self._predicate), self._replacement_value)
        return x.where(self._predicate, self._replacement_value)


def _not(predicate: MaskingSource) -> MaskingSource:
    if callable(predicate):
        return lambda v: ~predicate(v)
    return ~(predicate.astype(np.bool_))
