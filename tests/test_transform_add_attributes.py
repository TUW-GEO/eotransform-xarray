from typing import Dict

import numpy as np
from xarray import DataArray

from assertions import assert_data_array_identical
from eotransform_xarray.transformers.add_attrs import AddAttrs
from factories import make_raster


def test_add_attributes_to_data_array():
    attributed = AddAttrs(dict(added='attribute'))(make_raster(np.full((8, 8), 42), attrs=dict(some='attribute')))
    assert_data_array_identical(attributed,
                                make_raster(np.full((8, 8), 42), attrs=dict(some='attribute', added='attribute')))


def test_add_new_attribute_based_on_xarray():
    def the_answer(x: DataArray) -> Dict:
        return dict(the_answer_is=x[0, 0].to_numpy()[0])

    attributed = AddAttrs(the_answer)(make_raster(np.full((8, 8), 42)))
    assert_data_array_identical(attributed, make_raster(np.full((8, 8), 42), attrs=dict(the_answer_is=42)))
