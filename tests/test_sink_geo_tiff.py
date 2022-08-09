import rasterio
import rioxarray

from assertions import assert_raster_eq
from eotransform_xarray.sinks.geo_tiff import SinkToGeoTiff
from factories import make_raster


def test_write_raster_as_geo_tiff_following_passed_naming_convention(tmp_path):
    sink = SinkToGeoTiff(out_dir=tmp_path, namer=lambda i, a: f"{a.name}_{i}.tif")
    for raster in [make_raster([[42, 42],
                                [42, 42]], name="test-raster"),
                   make_raster([[43, 43],
                                [43, 43]], name="test-raster")]:
        sink(raster)

    assert_raster_geo_tif_eq(tmp_path / "test-raster_0.tif", [[42, 42],
                                                              [42, 42]],
                             expected_attrs=dict(add_offset=0.0, long_name="test-raster", scale_factor=1.0))
    assert_raster_geo_tif_eq(tmp_path / "test-raster_1.tif", [[43, 43],
                                                              [43, 43]],
                             expected_attrs=dict(add_offset=0.0, long_name="test-raster", scale_factor=1.0))


def assert_raster_geo_tif_eq(tif_file, expected_array_data, expected_attrs):
    with rasterio.open(tif_file) as rds:
        array = rioxarray.open_rasterio(rds)
        array.attrs = {**array.attrs, **rds.tags()}
    assert_raster_eq(array, expected_array_data)
    assert array.attrs == expected_attrs
