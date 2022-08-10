from typing import Tuple, Sequence

import numpy as np
from eotransform.protocol.transformer import Transformer
from more_itertools import repeatfunc

from eotransform_xarray.transformers import XArrayData


class RandomSample(Transformer[XArrayData, Sequence[XArrayData]]):
    def __init__(self, n, ignore_extent=None):
        self._n = n
        self._ignore_extent_name = ignore_extent

    def __call__(self, x: XArrayData) -> Sequence[XArrayData]:
        patch_size = x.attrs['patch_size']
        extent = x.sizes['y'], x.sizes['x']

        has_extent_to_ignore = self._ignore_extent_name is not None
        if has_extent_to_ignore:
            ignore_extent = x.attrs['extents'][self._ignore_extent_name]
            valid_points = _find_valid_indices(patch_size, extent, ignore_extent)

        def _random_indices():
            if has_extent_to_ignore:
                i, j = valid_points[:, np.random.randint(valid_points.shape[1])]
            else:
                i = np.random.randint(0, extent[0] - patch_size)
                j = np.random.randint(0, extent[1] - patch_size)
            return i, j

        return [_sample_patch_from(x, position, patch_size) for position in repeatfunc(_random_indices, self._n)]


def _find_valid_indices(patch_size, extent, ignore_extent):
    ii, jj = np.meshgrid(range(0, extent[0] - patch_size), range(0, extent[1] - patch_size), indexing='ij')
    mask = np.zeros_like(ii, dtype=bool)
    left, top, right, bottom = ignore_extent
    left = max(left - patch_size, 0)
    top = max(top - patch_size, 0)
    right = min(right, mask.shape[0])
    bottom = min(bottom, mask.shape[1])
    mask[left:right, top:bottom] = True
    ii = np.ma.masked_array(ii, mask=mask).compressed()
    jj = np.ma.masked_array(jj, mask=mask).compressed()
    return np.stack([ii, jj])


def _sample_patch_from(x: XArrayData, position: Tuple[int, int], patch_size):
    top, left = position
    bottom = top + patch_size
    right = left + patch_size
    patch = x.isel(y=slice(top, bottom), x=slice(left, right))
    patch.attrs = patch.attrs.copy()
    patch.attrs['sampled_extent'] = (top, left, bottom, right)
    return patch
