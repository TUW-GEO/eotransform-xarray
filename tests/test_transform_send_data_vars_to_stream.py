import pytest
from eotransform.protocol.stream import StreamIn
from xarray import Dataset

from eotransform_xarray.transformers.send_dataset_to_stream import SendDataVarsToStream, DataVars
from factories import make_raster


class StreamInSpy(StreamIn[DataVars]):
    def __init__(self):
        self.received = None

    def send(self, data_vars: DataVars) -> None:
        self.received = data_vars


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
    sender = SendDataVarsToStream(stream, 'var_0', 'var_2')
    sender(src_ds)
    assert stream.received[0] == src_ds['var_0']
    assert stream.received[1] == src_ds['var_2']


def test_return_source_data_set(stream, src_ds):
    sender = SendDataVarsToStream(stream, 'var_0', 'var_2')
    returned = sender(src_ds)
    assert returned is src_ds


def test_preprocess_data_before_sending_to_stream(stream, src_ds):
    sender = SendDataVarsToStream(stream, 'var_0', 'var_2', preprocess=lambda x: x + 1)
    sender(src_ds)
    assert stream.received[0] == src_ds['var_0'] + 1
    assert stream.received[1] == src_ds['var_2'] + 1
