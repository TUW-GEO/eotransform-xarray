from enum import Enum, auto
from typing import Optional, Dict

import numpy as np
from xarray import DataArray

from eotransform_xarray.geometry.degrees import Degree
from eotransform_xarray.numba_engine.normalize_sig0_to_ref_lia_by_slope import normalize_numba
from eotransform_xarray.transformers import TransformerOfDataArray

ORBIT_KEY = 'orbit'


class Engine(Enum):
    DASK = auto()
    NUMBA = auto()


class NormalizeSig0ToRefLiaBySlope(TransformerOfDataArray):
    class MissingLiaError(KeyError):
        ...

    class MissingOrbitInfoError(RuntimeError):
        ...

    def __init__(self, slope: DataArray, lias_per_orbit: Dict[str, DataArray], reference_lia: Degree,
                 engine: Optional[Engine] = Engine.DASK):
        self._slope = slope
        self._lias_per_orbit = lias_per_orbit
        self._reference_lia = reference_lia
        self._engine = engine

    def __call__(self, x: DataArray) -> DataArray:
        if ORBIT_KEY not in x.attrs:
            raise self.MissingOrbitInfoError(f"The input array doesn't have orbit information it its attrs: {x.attrs}")
        orbit = x.attrs[ORBIT_KEY]
        if orbit not in self._lias_per_orbit:
            raise self.MissingLiaError(f"No LIA map found for orbit {orbit}.")

        lia = self._lias_per_orbit[orbit]
        if self._engine == Engine.DASK:
            return x - self._slope * (lia - self._reference_lia.value)
        if self._engine == Engine.NUMBA:
            out = np.empty_like(x.values)
            normalize_numba(x.values, self._slope.values, lia.values, self._reference_lia.value,
                            out)
            return x.copy(data=out)
