import pytest
from xarray import Dataset

from eotransform_xarray.transformers.send_dataset_to_stream import StreamIn, SendDatasetToStream
from factories import make_raster


class StreamInSpy(StreamIn):
    def __init__(self):
        self.received = None

    def send(self, *args):
        self.received = args


@pytest.fixture
def stream():
    return StreamInSpy()


@pytest.fixture
def src_ds():
    return Dataset(dict(
        var_0=make_raster([[1]]),
        var_1=make_raster([[2]]),
        var_2=make_raster([[3]]),
    ))


def test_send_dataset_vars_to_stream(stream, src_ds):
    sender = SendDatasetToStream(stream, 'var_0', 'var_2')
    sender(src_ds)
    assert stream.received[0] == src_ds['var_0']
    assert stream.received[1] == src_ds['var_2']


def test_return_source_data_set(stream, src_ds):
    sender = SendDatasetToStream(stream, 'var_0', 'var_2')
    returned = sender(src_ds)
    assert returned is src_ds


def test_preprocess_data_before_sending_to_stream(stream, src_ds):
    sender = SendDatasetToStream(stream, 'var_0', 'var_2', preprocess=lambda x: x + 1)
    sender(src_ds)
    assert stream.received[0] == src_ds['var_0'] + 1
    assert stream.received[1] == src_ds['var_2'] + 1
