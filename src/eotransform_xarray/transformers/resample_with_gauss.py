from dataclasses import dataclass, asdict
from typing import Tuple, Optional

import numpy as np
import rioxarray  # noqa # pylint: disable=unused-import
import xarray as xr
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
                 raster_chunk_sizes: Tuple[int, int], n_procs: Optional[int] = 1,
                 resampling_parameter_storage: Optional[Storage] = None):
        self._area_dst = area_dst
        self._params_storage = resampling_parameter_storage or StorageIntoTheVoid()
        if self._params_storage.exists():
            self._projection_params = ProjectionParameter.from_storage(self._params_storage)
        else:
            self._projection_params = self._calc_projection(swath_src, area_dst, neighbours, lookup_radius,
                                                            raster_chunk_sizes, n_procs)
            self._projection_params.store(self._params_storage)
        self._projection_params.out_resampling['weights'] = \
            self._distances_to_gauss_weights(self._projection_params.out_resampling['weights'], sigma)

    @staticmethod
    def _calc_projection(swath: Swath, area: Area, neighbours: int, lookup_radius: float,
                         raster_chunks: Tuple[int, int], n_procs: int) -> ProjectionParameter:
        sw_def = SwathDefinition(swath.lons.swapaxes(0, -1), swath.lats.swapaxes(0, -1))
        ar_def = AreaDefinition(area.name, area.description, "proj_id", area.projection, area.columns, area.rows,
                                area.extent.to_tuple())
        val_in_idc, val_out_idc, idc, distances = get_neighbour_info(sw_def, ar_def, lookup_radius, neighbours,
                                                                     nprocs=n_procs)
        packed_idc = MaybePacked(idc) | np.uint8 | np.uint16 | np.uint32 | np.uint64
        packed_idc = packed_idc.value.swapaxes(0, -1).reshape((-1, area.rows, area.columns))
        distances = distances.swapaxes(0, -1).astype(np.float32).reshape((-1, area.rows, area.columns))
        out_mask = val_out_idc[np.newaxis, ...].reshape((-1, area.rows, area.columns))
        return ProjectionParameter(Dataset({'mask': (('cell', 'location'), val_in_idc[np.newaxis, ...])},
                                           coords={'lon': ('location', swath.lons[0]),
                                                   'lat': ('location', swath.lats[0])})
                                   .chunk({'cell': -1, 'location': -1}),
                                   Dataset({'indices': (('neighbours', 'y', 'x'), packed_idc),
                                            'weights': (('neighbours', 'y', 'x'), distances),
                                            'mask': (('cell', 'y', 'x'), out_mask)})
                                   .chunk({'neighbours': -1, 'cell': -1, 'y': raster_chunks[0], 'x': raster_chunks[1]})
                                   .rio.write_crs(area.projection).rio.write_transform(area.transform))

    @staticmethod
    def _distances_to_gauss_weights(distances: DataArray, sigma: float) -> DataArray:
        sig_sqrd = sigma ** 2
        return np.exp(-distances ** 2 / sig_sqrd)

    def __call__(self, x: DataArray) -> DataArray:
        self._sanity_check_input(x)
        r_arr = xr.apply_ufunc(_resample,
                               self._projection_params.out_resampling['indices'],
                               self._projection_params.out_resampling['weights'],
                               self._projection_params.out_resampling['mask'][0].astype(bool),
                               x,
                               self._projection_params.in_resampling['mask'][0].astype(bool),
                               input_core_dims=[['neighbours'], ['neighbours'], [], x.dims[-1:], ['location']],
                               output_dtypes=[np.float32],
                               dask='parallelized', keep_attrs=True)
        r_arr = r_arr.transpose(*x.dims[:2], 'y', 'x')
        crds = {c: x.coords[c] for c in x.coords if c in x.dims and c not in {'y', 'x'}}
        r_arr = r_arr.assign_coords(crds)
        r_arr.attrs = x.attrs
        return r_arr

    def _sanity_check_input(self, x: DataArray):
        if self._projection_params.in_resampling['mask'].size != x.shape[-1]:
            raise ResampleWithGauss.MismatchError("Mismatch between resample transformation projection and input data:"
                                                  "\nvalid_indices' size doesn't match input data value length:\n"
                                                  f"{self._projection_params.in_resampling.sizes} != {x.shape}")


def _resample(indices: NDArray, weights: NDArray, out_valid: NDArray, in_data: NDArray, in_valid: NDArray) -> NDArray:
    in_data = np.ma.array(in_data, mask=np.isnan(in_data) | ~in_valid[np.newaxis, np.newaxis, :])
    times, parameters, in_size = in_data.shape
    indices = indices.squeeze((2, 3))
    out_valid = out_valid.squeeze(3)
    invalid_idc = indices == in_size
    out = in_data[:, :, np.where(invalid_idc, 0, indices)]
    out[:, :, ~out_valid | invalid_idc] = np.ma.masked

    weights = weights.squeeze((2, 3))
    out, w_sums = np.ma.average(out, axis=-1, weights=np.broadcast_to(weights, (times, parameters) + weights.shape),
                                returned=True)
    out[w_sums <= 0] = np.ma.masked
    return out.filled(np.nan).transpose((2, 3, 0, 1))
