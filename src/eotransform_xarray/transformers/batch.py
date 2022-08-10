from typing import Sequence

from eotransform.protocol.transformer import Transformer

from eotransform_xarray.transformers import XArrayData
from eotransform_xarray.transformers.files_to_xarray import CONCATED_ATTRS_KEY

BATCHED_ATTRS_KEY = 'batched_attrs'


class Batch(Transformer[XArrayData, Sequence[XArrayData]]):
    def __init__(self, size: int, batch_dim: str):
        self._size = size
        self._batch_dim = batch_dim

    def __call__(self, x: XArrayData) -> Sequence[XArrayData]:
        n_samples = x.sizes[self._batch_dim]
        batches = [x.isel({self._batch_dim: slice(i, min(i + self._size, n_samples))})
                   for i in range(0, n_samples, self._size)]
        if CONCATED_ATTRS_KEY in x.attrs:
            self._split_concated_attributes_to_batches(x, batches, n_samples)
        return batches

    def _split_concated_attributes_to_batches(self, x, batches, n_samples):
        concated_attrs = x.attrs[CONCATED_ATTRS_KEY]
        n_concated = len(concated_attrs)
        assert n_concated == n_samples, f"length of concatenated attributes {n_concated}," \
                                        f" must match length of input array {n_samples}"
        split_attrs = [concated_attrs[i:min(i + self._size, n_concated)] for i in range(0, n_concated, self._size)]
        for batch, attr in zip(batches, split_attrs):
            batch.attrs = batch.attrs.copy()
            batch.attrs[BATCHED_ATTRS_KEY] = attr
            del batch.attrs[CONCATED_ATTRS_KEY]
