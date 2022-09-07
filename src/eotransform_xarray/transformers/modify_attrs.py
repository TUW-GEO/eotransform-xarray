from typing import Callable, Dict

from eotransform_xarray.transformers import TransformerOfXArrayData, XArrayData


class ModifyAttrs(TransformerOfXArrayData):
    def __init__(self, modification_fn: Callable[[Dict], Dict]):
        self._modification_fn = modification_fn

    def __call__(self, x: XArrayData) -> XArrayData:
        x.attrs = self._modification_fn(x.attrs)
        return x
