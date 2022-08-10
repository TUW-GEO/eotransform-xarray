import numpy as np
import pytest

from eotransform_xarray.transformers.batch import BATCHED_ATTRS_KEY, Batch
from eotransform_xarray.transformers.files_to_xarray import CONCATED_ATTRS_KEY
from factories import make_raster


@pytest.fixture(params=['data_array', 'dataset'])
def make_raster_source(request):
    def _f(values, attrs=None):
        r = make_raster(values, attrs=attrs)
        if request.param == 'dataset':
            return r.to_dataset(name='raster', promote_attrs=True)
        return r

    return _f


def test_batches_of_specified_size(make_raster_source):
    source = make_raster_source(np.zeros((16, 8, 8)))
    batched = Batch(size=4, batch_dim='band')(source)
    assert_batch_sizes(batched, [4, 4, 4, 4])


def assert_batch_sizes(actual_batches, sizes):
    for batch, size in zip(actual_batches, sizes):
        assert batch.sizes['band'] == size


def test_handle_tail_correctly(make_raster_source):
    source = make_raster_source(np.zeros((16, 8, 8)))
    batched = Batch(size=5, batch_dim='band')(source)
    assert_batch_sizes(batched, [5, 5, 5, 1])


def test_batch_concatenated_attributes(make_raster_source):
    source = make_raster_source(np.zeros((16, 8, 8)), attrs={CONCATED_ATTRS_KEY: [dict(some=i) for i in range(16)]})
    batched = Batch(size=5, batch_dim='band')(source)
    assert_batched_attributes(batched, [[dict(some=i) for i in range(5)],
                                        [dict(some=i) for i in range(5, 10)],
                                        [dict(some=i) for i in range(10, 15)],
                                        [dict(some=15)]])


def assert_batched_attributes(actual_batches, attributes):
    for batch, attr in zip(actual_batches, attributes):
        assert batch.attrs[BATCHED_ATTRS_KEY] == attr
        assert CONCATED_ATTRS_KEY not in batch.attrs
