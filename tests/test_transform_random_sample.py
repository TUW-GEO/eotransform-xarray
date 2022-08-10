import numpy as np
import pytest

from eotransform_xarray.transformers.random_sample import RandomSample
from factories import make_raster


@pytest.fixture(params=['data_array', 'dataset'])
def make_random_sample_source_per_data_type(request):
    def _f(with_extent=None):
        attrs = dict(patch_size=8)
        if with_extent:
            attrs['extents'] = with_extent

        raster = make_raster(np.random.randn(6, 32, 32), attrs=attrs)
        if request.param == 'dataset':
            return raster.to_dataset(name='raster', promote_attrs=True)
        return raster

    return _f, request.param


@pytest.fixture
def make_random_sample_source(make_random_sample_source_per_data_type):
    f, _ = make_random_sample_source_per_data_type
    return f


@pytest.fixture
def fixed_seed_sample_extents(make_random_sample_source_per_data_type, fixed_seed):
    _, data_type = make_random_sample_source_per_data_type
    extent_for_datatype = {
        'data_array': [(18, 13, 26, 21), (21, 12, 29, 20)],
        'dataset': [(20, 1, 28, 9), (23, 3, 31, 11)],
    }
    return extent_for_datatype[data_type]


def test_sample_specified_amount(make_random_sample_source):
    raster = make_random_sample_source()
    samples = RandomSample(n=3)(raster)
    assert len(samples) == 3


def test_sample_of_configured_patch_size(make_random_sample_source):
    raster = make_random_sample_source()
    sample = RandomSample(n=1)(raster)[0]
    assert sample.sizes == dict(band=6, y=8, x=8)


def test_samples_are_random(make_random_sample_source):
    raster = make_random_sample_source()
    samples = RandomSample(n=3)(raster)
    assert_are_random(samples)


def assert_are_random(actual_samples):
    is_different = False
    for lhs in actual_samples:
        for rhs in actual_samples:
            if np.not_equal(lhs.values, rhs.values).any():
                is_different = True
                break
    assert is_different


@pytest.mark.parametrize("extent", [(4, 4, 15, 15),
                                    (0, 24, 32, 32)])
def test_ignore_extent(extent, make_random_sample_source):
    raster = make_random_sample_source(with_extent=dict(validation=extent))
    samples = RandomSample(n=32, ignore_extent='validation')(raster)
    y, x, yy, xx = extent
    assert_samples_do_not_contain(samples, raster.isel(y=slice(y, yy), x=slice(x, xx)))


def assert_samples_do_not_contain(samples, expected_ignored_area):
    overlaps = False
    for sample in samples:
        if is_overlapping(sample.values, expected_ignored_area.values):
            overlaps = True
            break

    assert not overlaps


def is_overlapping(lhs: np.ndarray, rhs: np.ndarray) -> bool:
    return np.any(np.isin(lhs, rhs))


def test_add_sample_extent_to_attributes(make_random_sample_source, fixed_seed_sample_extents):
    raster = make_random_sample_source()
    samples = RandomSample(n=2)(raster)
    assert samples[0].attrs['sampled_extent'] == fixed_seed_sample_extents[0]
    assert samples[1].attrs['sampled_extent'] == fixed_seed_sample_extents[1]
