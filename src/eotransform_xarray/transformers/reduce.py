from abc import abstractmethod
from typing import Protocol, Callable, Optional

from eotransform_xarray.transformers import TransformerOfDataArray
from numpy._typing import NDArray
from xarray import DataArray


class ReducerFn(Protocol):
    @abstractmethod
    def __call__(self, array: NDArray, axis: int, **kwargs) -> NDArray:
        ...


class Reduce(TransformerOfDataArray):
    def __init__(self, func: Callable, dim: Optional[str] = None, *, axis: Optional[int] = None,
                 keep_attrs: Optional[bool] = None, keep_dims: Optional[bool] = False, **kwargs):
        self._func = func
        self._dim = dim
        self._axis = axis
        self._keep_attrs = keep_attrs
        self._keep_dims = keep_dims
        self._kwargs = kwargs

    def __call__(self, x: DataArray) -> DataArray:
        return x.reduce(self._func, self._dim,
                        axis=self._axis, keep_attrs=self._keep_attrs, keepdims=self._keep_dims, **self._kwargs)
