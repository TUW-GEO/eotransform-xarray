from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import rioxarray
from eotransform.streamed_process import streamed_process
from eotransform.transformers.compose import Compose

from eotransform_xarray.sinks.geo_tiff import SinkToGeoTiff
from eotransform_xarray.transformers.masking_where import MaskWhere
from eotransform_xarray.transformers.resample_with_gauss import ResampleWithGauss
from eotransform_xarray.transformers.squeeze import Squeeze
from factories import make_swath, make_swath_data_array, make_target_area


@pytest.fixture
def dst_dir(tmp_path):
    d = tmp_path / 'dst'
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_example_doc_streamed_resample_and_mask(dst_dir):
    swath_geometry = make_swath([12.0, 16.0], [47.9, 45.2])
    input_src = [
        make_swath_data_array([[[1, 2, 4, 8]], [[1, 2, 4, np.nan]]], swath_geometry),
        make_swath_data_array([[[3, 2, 1, 2]], [[3, 4, 3, 2]]], swath_geometry),
    ]

    raster_geometry = make_target_area(200, 200)

    # begin-snippet: streamed_resample_and_mask
    resample = ResampleWithGauss(swath_geometry, raster_geometry, sigma=2e5, neighbours=4, lookup_radius=1e6)
    mask = MaskWhere(lambda x: x > 2, np.nan)
    squeeze = Squeeze()
    with ThreadPoolExecutor(max_workers=3) as ex:
        pipeline = Compose([resample, mask, squeeze])
        streamed_process(input_src, pipeline, SinkToGeoTiff(dst_dir, lambda i, da: f"out_{i}.tif"), ex)
    # end-snippet

    assert (dst_dir / "out_0.tif").exists()
    assert (dst_dir / "out_1.tif").exists()
