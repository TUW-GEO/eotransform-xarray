from typing import Dict, Union, Callable, Sequence, Tuple, Optional

import xarray as xr
from numpy._typing import NDArray
from xarray import DataArray

from eotransform_xarray.transformers import AggregatorOfDataArrays


class AggregationOutput:
    def __init__(self, dim_sizes, dtype):
        self.dims = []
        self.sizes = []

        for d, s in dim_sizes:
            self.dims.append(d)
            self.sizes.append(s)
        self.dtype = dtype

    def get_sizes_dict(self) -> Dict[str, int]:
        return {d: s for d, s in zip(self.dims, self.sizes)}


class AggregateAlongDim(AggregatorOfDataArrays):
    def __init__(self, dim: Union[int, str], aggregate: Callable[..., NDArray],
                 output: Optional[Sequence[AggregationOutput]] = None,
                 kwargs: Optional[Dict] = None):
        self._dim = dim
        self._aggregate = aggregate
        self._output = output or []
        self._kwargs = kwargs or {}

    def __call__(self, x: Tuple[DataArray, ...]) -> DataArray:
        input_cores = [[a.dims[self._dim]] for a in x]
        return xr.apply_ufunc(self._aggregate, *x, kwargs=self._kwargs, input_core_dims=input_cores,
                              output_core_dims=[o.dims for o in self._output], dask='parallelized',
                              output_dtypes=[o.dtype for o in self._output],
                              dask_gufunc_kwargs={'output_sizes': {k: v for o in self._output
                                                                   for k, v in o.get_sizes_dict().items()}})
