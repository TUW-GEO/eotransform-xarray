from typing import Union, Dict, Callable

from eotransform_xarray.transformers import TransformerOfDataArray, XArrayData


class AddAttrs(TransformerOfDataArray):
    def __init__(self, extra_attrs: Union[Dict, Callable[[XArrayData], Dict]]):
        self._extra_attrs = extra_attrs

    def __call__(self, x: XArrayData) -> XArrayData:
        if callable(self._extra_attrs):
            ea = self._extra_attrs(x)
        else:
            ea = self._extra_attrs
        return x.assign_attrs(ea)
