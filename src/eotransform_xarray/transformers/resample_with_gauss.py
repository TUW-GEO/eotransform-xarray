from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Mapping, Any

import dask.array as da
import numpy as np
import xarray as xr
import rioxarray  # noqa # pylint: disable=unused-import
from affine import Affine
from numpy.typing import NDArray, DTypeLike
from xarray import DataArray, Dataset

from eotransform_xarray.storage.storage import Storage

try:
    from numba import njit, prange
    from pyresample import SwathDefinition, AreaDefinition
    from pyresample.kd_tree import get_neighbour_info
except ImportError:
    print("ResampleWithGauss requires numba and pyresample.\npip install numba pyresample")
    raise

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
    in_resampling: Dataset
    out_resampling: Dataset

    @classmethod
    def from_storage(cls, storage: Storage) -> "ProjectionParameter":
        return ProjectionParameter(**{f: v for f, v in storage.load().items()})

    def store(self, storage: Storage) -> None:
        storage.save(asdict(self))


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


class StorageIntoTheVoid(Storage):
    def exists(self) -> bool:
        return False

    def load(self) -> Dataset:
        raise NotImplementedError("Can't load from the void.")

    def save(self, data: Dataset) -> None:
        pass


class ResampleWithGauss(TransformerOfDataArray):
    class MismatchError(ValueError):
        ...

    def __init__(self, swath_src: Swath, area_dst: Area, sigma: float, neighbours: int, lookup_radius: float,
                 n_procs: Optional[int] = 1, resampling_parameter_storage: Optional[Storage] = None):
        self._area_dst = area_dst
        self._params_storage = resampling_parameter_storage or StorageIntoTheVoid()
        if self._params_storage.exists():
            self._projection_params = ProjectionParameter.from_storage(self._params_storage)
        else:
            self._projection_params = self._calc_projection(swath_src, area_dst, neighbours, lookup_radius, n_procs)
            self._projection_params.store(self._params_storage)
        self._transform_distances_to_gauss_weights(self._projection_params.out_resampling['weights'].values, sigma)

    @staticmethod
    def _calc_projection(swath: Swath, area: Area, neighbours: int, lookup_radius: float,
                         n_procs: int) -> ProjectionParameter:
        sw_def = SwathDefinition(swath.lons.swapaxes(0, -1), swath.lats.swapaxes(0, -1))
        ar_def = AreaDefinition(area.name, area.description, "proj_id", area.projection, area.columns, area.rows,
                                area.extent.to_tuple())
        val_in_idc, val_out_idc, idc, distances = get_neighbour_info(sw_def, ar_def, lookup_radius, neighbours,
                                                                     nprocs=n_procs)
        packed_idc = MaybePacked(idc) | np.uint8 | np.uint16 | np.uint32 | np.uint64
        packed_idc = packed_idc.value.swapaxes(0, -1)
        distances = distances.swapaxes(0, -1).astype(np.float32)
        out_mask = val_out_idc[np.newaxis, ...]
        return ProjectionParameter(Dataset({'mask': val_in_idc}, coords={'lon': ('location', swath.lons[0]),
                                                                         'lat': ('location', swath.lats[0])}),
                                   ResampleWithGauss._to_raster_dataset(
                                       {'indices': (('neighbours', 'location'), packed_idc),
                                        'weights': (('neighbours', 'location'), distances),
                                        'mask': (('cell', 'location'), out_mask)},
                                       area))

    @staticmethod
    def _to_raster_dataset(datas: Mapping[str, Tuple[Tuple[str, str], NDArray]], area: Area) -> Dataset:
        y = np.tile(np.arange(200), 40000 // 200)
        x = np.tile(np.arange(200), 40000 // 200)
        ds = Dataset(datas, coords={'y': ('location', y), 'x': ('location', x)})
        ds = ds.set_index(location=("y", "x")).unstack("location")
        ds.rio.set_spatial_dims('x', 'y', inplace=True)
        ds.rio.write_crs(area.projection, inplace=True)
        ds.rio.write_transform(area.transform, inplace=True)
        return ds

    @staticmethod
    def _transform_distances_to_gauss_weights(distances: NDArray, sigma: float) -> None:
        gauss_parallel_inplace(distances, sigma)

    def __call__(self, x: DataArray) -> DataArray:
        self._sanity_check_input(x)
        r_arr = xr.apply_ufunc(_resample, x,
                               self._projection_params.valid_input,
                               self._projection_params.indices,
                               self._projection_params.weights,
                               self._projection_params.valid_output,
                               input_core_dims=[['value', 'time', 'parameter'], [], [], [], []],
                               output_core_dims=[['value', 'time', 'parameter']],
                               exclude_dims={'value'},
                               dask='parallelized')

        # r_arr = r_arr.assign_coords({'y': ('value', np.arange(r_arr.sizes['value'])),
        #                              'x': ('value', np.arange(r_arr.sizes['value']))})
        # r_arr = r_arr.set_index(value=('y', 'x')).unstack('value')
        r_arr.rio.write_crs(self._area_dst.projection, inplace=True)
        r_arr.rio.write_transform(self._area_dst.transform, inplace=True)
        crds = {c: x.coords[c] for c in x.coords if c in x.dims and c not in {'y', 'x'}}
        r_arr = r_arr.assign_coords(crds)
        return r_arr

    def _numba_resample(self, x):
        valid_data = x[..., self._projection_params.valid_input]
        result = np.empty((valid_data.shape[0], valid_data.shape[1], self._projection_params.indices.shape[-1]),
                          dtype=np.float32)
        _resample_swath_to_area(self._projection_params.indices,
                                self._projection_params.weights, valid_data.values,
                                self._projection_params.valid_output,
                                result)
        result = result.reshape((result.shape[0], result.shape[1], self._area_dst.rows, self._area_dst.columns))
        r_arr = DataArray(result, dims=(*x.dims[:-1], "y", "x"), attrs=x.attrs)
        r_arr.rio.write_crs(self._area_dst.projection, inplace=True)
        r_arr.rio.write_transform(self._area_dst.transform, inplace=True)
        crds = {c: x.coords[c] for c in x.coords if c in x.dims and c not in {'y', 'x'}}
        r_arr = r_arr.assign_coords(crds)
        return r_arr

    def _sanity_check_input(self, x: DataArray):
        if self._projection_params.valid_input.size != x.shape[-1]:
            raise ResampleWithGauss.MismatchError("Mismatch between resample transformation projection and input data:"
                                                  "\nvalid_indices' size doesn't match input data value length:\n"
                                                  f"{self._projection_params.valid_input.shape} != {x.shape}")


def _resample(x, in_valid, indices, weights, out_valid):
    x = x[in_valid, ...]
    neighbours, out_size = indices.shape
    in_size, times, parameters = x.shape
    out = np.empty((out_size, times, parameters), dtype=np.float32)

    for parameter in range(parameters):
        for time in range(times):
            for out_idx in range(out_size):
                if out_valid[out_idx]:
                    weighted_sum = 0.0
                    sum_of_weights = 0.0
                    for n_i in range(neighbours):
                        sample_idx = indices[n_i, out_idx]
                        if sample_idx != in_size:
                            weight = weights[n_i, out_idx]
                            sampled = x[sample_idx, time, parameter]
                            if not np.isnan(sampled):
                                weighted_sum += sampled * weight
                                sum_of_weights += weight

                    out[out_idx, time, parameter] = weighted_sum / sum_of_weights if sum_of_weights > 0 else np.nan
                else:
                    out[out_idx, time, parameter] = np.nan

    return out


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
