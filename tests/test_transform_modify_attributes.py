import numpy as np
import pytest

from eotransform_xarray.transformers.modify_attrs import ModifyAttrs
from factories import make_raster


@pytest.fixture(params=['data_array', 'dataset'])
def make_xdata(request):
    def _fn(attrs):
        raster = make_raster(np.zeros((1, 8, 8)), attrs=attrs)
        if request.param == 'dataset':
            return raster.to_dataset(name='raster', promote_attrs=True)
        return raster

    return _fn


def test_modify_attributes_with_given_function(make_xdata):
    modified = ModifyAttrs(lambda attrs: {**attrs, **dict(another=1)})
    assert modified(make_xdata(dict(some='attribute'))).attrs == dict(some='attribute', another=1)
