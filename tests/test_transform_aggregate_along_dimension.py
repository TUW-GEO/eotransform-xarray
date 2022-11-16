import numpy as np

from eotransform_xarray.functional.stats import pearsonr
from eotransform_xarray.transformers.aggregate_along_dim import AggregationOutput, AggregateAlongDim
from factories import make_raster


def test_aggregate_pearson_r_along_first_dimension():
    correlate = AggregateAlongDim(dim=0, aggregate=pearsonr, output=[AggregationOutput([('stats', 3)], float)])

    random_array = make_randn_ts((500, 20, 20), scale=1.0, bias=10)
    source = make_raster(random_array).chunk((-1, 10, 10))

    target = make_raster(random_array + make_randn_ts((500, 20, 20), scale=0.1, bias=100)).chunk((-1, 10, 10))
    r = correlate((source, target))
    assert np.mean(r.values[0]) >= 0.9 and r.sizes == {'y': 20, 'x': 20, 'stats': 3}

    target = make_raster(random_array + make_randn_ts((500, 20, 20), scale=10.0, bias=10)).chunk((-1, 10, 10))
    r = correlate((source, target))
    assert 0.2 >= np.mean(r.values[0]) >= 0.0 and r.sizes == {'y': 20, 'x': 20, 'stats': 3}


def make_randn_ts(shape, scale=1.0, bias=0):
    return (np.random.randn(*shape) * scale + bias).astype(np.float32)
