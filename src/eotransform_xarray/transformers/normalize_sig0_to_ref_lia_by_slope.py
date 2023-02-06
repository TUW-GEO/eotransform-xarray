from copy import copy
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Dict

import numpy as np
from xarray import DataArray

from eotransform_xarray.constants import SOURCE_KEY
from eotransform_xarray.functional.load_file_dataframe_to_array import FILEPATH_COORD
from eotransform_xarray.geometry.degrees import Degree
from eotransform_xarray.numba_engine.normalize_sig0_to_ref_lia_by_slope import normalize_numba
from eotransform_xarray.transformers import TransformerOfDataArray

ORBIT_COORD = 'orbit'
METADATA_KEY = 'normalize_sig0_to_ref_lia_by_slope_meta'
SLOPE_SRC_KEY = 'slope_source'
LIA_SRC_KEY = 'lia_source'
SIG0_SRC_KEY = 'sig0_source'
REF_ANGLE_KEY = 'reference_angle'
ENGINE_USED_KEY = 'engine_used'


class Engine(Enum):
    DASK = auto()
    NUMBA = auto()


class NormalizeSig0ToRefLiaBySlope(TransformerOfDataArray):
    class MissingLiaError(KeyError):
        ...

    class MissingOrbitInfoError(RuntimeError):
        ...

    def __init__(self, slope: DataArray, lias_per_orbit: DataArray, reference_lia: Degree,
                 engine: Optional[Engine] = Engine.DASK):
        self._slope = slope
        self._lias_per_orbit = lias_per_orbit
        self._reference_lia = reference_lia
        self._engine = engine

    def __call__(self, x: DataArray) -> DataArray:
        if ORBIT_COORD not in x.coords:
            raise self.MissingOrbitInfoError(
                f"The input array doesn't have orbit information it its coords: {x.coords}")
        orbit = x.coords[ORBIT_COORD].values.item()
        if orbit not in self._lias_per_orbit.coords[ORBIT_COORD]:
            raise self.MissingLiaError(
                f"No LIA map found for orbit {orbit}, available orbits are {self._lias_per_orbit.coords[ORBIT_COORD]}")

        lia = self._lias_per_orbit.sel({ORBIT_COORD: orbit})
        if self._engine == Engine.DASK:
            nx = x - self._slope * (lia - self._reference_lia.value)
        elif self._engine == Engine.NUMBA:
            out = np.empty_like(x.values)
            normalize_numba(x.values, self._slope.values, lia.values, self._reference_lia.value,
                            out)
            nx = x.copy(data=out)
        else:
            raise NotImplementedError(f"engine {self._engine.name} currently not supported.")

        return self._populate_with_normalize_metadata(nx, x, lia)

    def _populate_with_normalize_metadata(self, normalized: DataArray, sig0: DataArray, lia: DataArray) -> DataArray:
        normalized.attrs = copy(sig0.attrs)
        normalized.attrs[METADATA_KEY] = {
            SLOPE_SRC_KEY: _try_get_filepath_for_array(self._slope),
            LIA_SRC_KEY: _try_get_filepath_for_array(lia),
            SIG0_SRC_KEY: _try_get_filepath_for_array(sig0),
            REF_ANGLE_KEY: self._reference_lia.value,
            ENGINE_USED_KEY: self._engine.name
        }

        return normalized


def _try_get_filepath_for_array(a: DataArray) -> str:
    if FILEPATH_COORD in a.coords:
        return str(a.coords[FILEPATH_COORD].values.item())
    else:
        return a.encoding.get(SOURCE_KEY,
                              f"source definition missing, either encoding['source'] or as {FILEPATH_COORD} coords.")
