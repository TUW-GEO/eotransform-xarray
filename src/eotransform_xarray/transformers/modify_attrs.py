from typing import Callable, Dict

from eotransform_xarray.transformers import TransformerOfXArrayData, XArrayData


class ModifyAttrs(TransformerOfXArrayData):
    """
    Modify the attributes of a xarray DataArray or Dataset with the given function
    >>> from xarray import DataArray
    >>> def add_foo_bar(attrs):
    ...     attrs['foo'] = "bar"
    ...     return attrs
    >>> ModifyAttrs(add_foo_bar)(DataArray([0], attrs=dict(initial='attribute'))).attrs
    {'initial': 'attribute', 'foo': 'bar'}
    """

    def __init__(self, modification_fn: Callable[[Dict], Dict]):
        """
        :param modification_fn: function which is used to modify the attributes of the Dataset or DataArray
        """
        self._modification_fn = modification_fn

    def __call__(self, x: XArrayData) -> XArrayData:
        x.attrs = self._modification_fn(x.attrs)
        return x
