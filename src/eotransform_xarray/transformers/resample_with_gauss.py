from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import rioxarray  # noqa # pylint: disable=unused-import
from affine import Affine
from numba import njit, prange
from numpy._typing import NDArray, DTypeLike
from pyresample import SwathDefinition, AreaDefinition
from pyresample.kd_tree import get_neighbour_info
from xarray import DataArray

from eotransform_xarray.transformers import TransformerOfDataArray


@dataclass
class Swath:
    lons: NDArray
    lats: NDArray


@dataclass
class Extent:
    lower_left_x: float
    lower_left_y: float
    upper_right_x: float
    upper_right_y: float

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return self.lower_left_x, self.lower_left_y, self.upper_right_x, self.upper_right_y


@dataclass
class Area:
    name: str
    projection: str
    columns: int
    rows: int
    extent: Extent
    transform: Affine
    description: str = ""


@dataclass
class ProjectionParameter:
    valid_input_indices: NDArray
    valid_output_indices: NDArray
    indices: NDArray
    weights: NDArray


class MaybePacked:
    def __init__(self, value: NDArray, is_packed: bool = False):
        self.value = value
        self._is_packed = is_packed
        self._max = value.max()

    def __or__(self, dtype: DTypeLike) -> "MaybePacked":
        if self._is_packed:
            return self

        if self._max <= np.iinfo(dtype).max:
            return MaybePacked(self.value.astype(dtype), True)
        else:
            return self


@njit(parallel=True)
def gauss_parallel_inplace(distances: NDArray, sigma: float) -> None:
    sig_sqrd = sigma ** 2
    for i in prange(distances.shape[1]):
        distances[:, i] = np.exp(-distances[:, i] ** 2 / sig_sqrd)


class ResampleWithGauss(TransformerOfDataArray):
    class MismatchError(ValueError):
        ...

    def __init__(self, swath_src: Swath, area_dst: Area, sigma: float, neighbours: int, lookup_radius: float,
                 n_procs: Optional[int] = 1):
        self._area_dst = area_dst
        self._projection_params = self._calc_projection(swath_src, area_dst, neighbours, lookup_radius, n_procs)
        self._transform_distances_to_gauss_weights(self._projection_params.weights, sigma)

    @staticmethod
    def _calc_projection(swath: Swath, area: Area, neighbours: int, lookup_radius: float,
                         n_procs: int) -> ProjectionParameter:
        sw_def = SwathDefinition(swath.lons.swapaxes(0, -1), swath.lats.swapaxes(0, -1))
        ar_def = AreaDefinition(area.name, area.description, "proj_id", area.projection, area.columns, area.rows,
                                area.extent.to_tuple())
        val_in_idc, val_out_idc, idc, distances = get_neighbour_info(sw_def, ar_def, lookup_radius, neighbours,
                                                                     nprocs=n_procs)
        packed_idc = MaybePacked(idc) | np.uint8 | np.uint16 | np.uint32 | np.uint64
        return ProjectionParameter(val_in_idc, val_out_idc, packed_idc.value.swapaxes(0, -1),
                                   distances.swapaxes(0, -1).astype(np.float32))

    @staticmethod
    def _transform_distances_to_gauss_weights(distances: NDArray, sigma: float) -> None:
        gauss_parallel_inplace(distances, sigma)

    def __call__(self, x: DataArray) -> DataArray:
        self._sanity_check_input(x)
        valid_data = x[..., self._projection_params.valid_input_indices]
        result = np.empty((valid_data.shape[0], valid_data.shape[1], self._projection_params.indices.shape[-1]),
                          dtype=np.float32)
        _resample_swath_to_area(self._projection_params.indices,
                                self._projection_params.weights, valid_data.values,
                                self._projection_params.valid_output_indices,
                                result)

        result = result.reshape((result.shape[0], result.shape[1], self._area_dst.rows, self._area_dst.columns))
        r_arr = DataArray(result, dims=(*x.dims[:-1], "y", "x"), attrs=x.attrs)
        r_arr.rio.write_crs(self._area_dst.projection, inplace=True)
        r_arr.rio.write_transform(self._area_dst.transform, inplace=True)
        return r_arr

    def _sanity_check_input(self, x: DataArray):
        if self._projection_params.valid_input_indices.size != x.shape[-1]:
            raise ResampleWithGauss.MismatchError("Mismatch between resample transformation projection and input data:"
                                                  "\nvalid_indices' size doesn't match input data value length:\n"
                                                  f"{self._projection_params.valid_input_indices.shape} != {x.shape}")


@njit(parallel=True)
def _resample_swath_to_area(indices: NDArray, weights: NDArray, valid_data: NDArray, out_valid: NDArray,
                            out: NDArray) -> None:
    neighbours, out_size = indices.shape
    times, parameters, in_size = valid_data.shape

    for parameter in range(parameters):
        for time in range(times):
            for out_idx in prange(out_size):
                if out_valid[out_idx]:
                    weighted_sum = 0.0
                    sum_of_weights = 0.0
                    for n_i in prange(neighbours):
                        sample_idx = indices[n_i, out_idx]
                        if sample_idx != in_size:
                            weight = weights[n_i, out_idx]
                            sampled = valid_data[time, parameter, sample_idx]
                            if not np.isnan(sampled):
                                weighted_sum += sampled * weight
                                sum_of_weights += weight

                    out[time, parameter, out_idx] = weighted_sum / sum_of_weights if sum_of_weights > 0 else np.nan
                else:
                    out[time, parameter, out_idx] = np.nan
