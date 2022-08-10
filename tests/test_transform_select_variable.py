from assertions import assert_raster_eq, assert_data_array_identical
from eotransform_xarray.transformers.select_variable import SelectVariable
from factories import make_raster, make_dataset


def test_select_a_variable_from_a_dataset():
    variable = make_raster([[0]], name='raster')
    dataset = make_dataset(dict(raster=variable))
    assert_raster_eq(SelectVariable('raster')(dataset), variable)


def test_combine_dataset_attributes_with_variable_according_to_option():
    variable = make_raster([[0]], name='raster', attrs=dict(data_array='attribute foo', same=42))
    dataset = make_dataset(dict(raster=variable), attrs=dict(dataset='attribute bar', same=43))

    selected_dropped = SelectVariable('raster', combine_dataset_attrs='drop_conflicts')(dataset)
    assert_data_array_identical(selected_dropped, make_raster([[0]], name='raster',
                                                              attrs=dict(data_array='attribute foo',
                                                                         dataset='attribute bar')))
