from enum import Enum, auto
from typing import Optional

import numpy as np
from xarray import DataArray

from eotransform_xarray.geometry.degrees import Degree
from eotransform_xarray.numba_engine.normalize_sig0_to_ref_lia_by_slope import normalize_numba
from eotransform_xarray.transformers import TransformerOfDataArray


class Engine(Enum):
    DASK = auto()
    NUMBA = auto()


class NormalizeSig0ToRefLiaBySlope(TransformerOfDataArray):
    def __init__(self, slope: DataArray, lia: DataArray, reference_lia: Degree, engine: Optional[Engine] = Engine.DASK):
        self._slope = slope
        self._lia = lia
        self._reference_lia = reference_lia
        self._engine = engine

    def __call__(self, x: DataArray) -> DataArray:
        if self._engine == Engine.DASK:
            return x - self._slope * (self._lia - self._reference_lia.value)
        if self._engine == Engine.NUMBA:
            out = np.empty_like(x.values)
            normalize_numba(x.values, self._slope.values, self._lia.values, self._reference_lia.value, out)
            return x.copy(data=out)
