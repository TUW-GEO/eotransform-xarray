![Coverage badge](https://raw.githubusercontent.com/TUW-GEO/eotransform-xarray/python-coverage-comment-action-data/badge.svg)
# eotransform-xarray

## What can I use eotransform-xarray for?

The eotransform-xarray package provides common transformations on raster data represented as [xarray](https://docs.xarray.dev/en/stable/) data structures, following the Transformer protocol of [eotransform](https://github.com/TUW-GEO/eotransform).
This makes them easy to mix and match, and you can quickly chain processing pipelines, using other [eotransform](https://github.com/TUW-GEO/eotransform) protocols.
Additionally, processing pipelines constructed from these Transformers, can be automatically applied to the `streamed_process` function from [eotransform](https://github.com/TUW-GEO/eotransform), to benefit from I/O hiding.

## Getting Started
### Installation
```bash
pip install eotransform-xarray
```

### Example: streamed processing pipeline
In the following example swath data is resampled, masked and written out as a GeoTIFF stack.

<!-- snippet: streamed_resample_and_mask -->
<a id='snippet-streamed_resample_and_mask'></a>
```py
resample = ResampleWithGauss(swath_geometry, raster_geometry, sigma=2e5, neighbours=4, lookup_radius=1e6)
mask = MaskWhere(lambda x: x > 2, np.nan)
squeeze = Squeeze()
with ThreadPoolExecutor(max_workers=3) as ex:
    pipeline = Compose([resample, mask, squeeze])
    streamed_process(input_src, pipeline, SinkToGeoTiff(dst_dir, lambda i, da: f"out_{i}.tif"), ex)
```
<sup><a href='/tests/test_doc_examples.py#L32-L39' title='Snippet source file'>snippet source</a> | <a href='#snippet-streamed_resample_and_mask' title='Start of snippet'>anchor</a></sup>
<!-- endSnippet -->

Note, that this example uses [eotransform](https://github.com/TUW-GEO/eotransform)'s `streamed_process` function to hide the I/O operations, using the compute resources more effectively.

### Dependencies:
eotransform-xarray requires Python 3.8 and has these dependencies:

<!-- snippet: dependencies -->
<a id='snippet-dependencies'></a>
```cfg
eotransform>=1.8
xarray
rioxarray
numpy
affine
more_itertools
```
<sup><a href='/setup.cfg#L29-L36' title='Snippet source file'>snippet source</a> | <a href='#snippet-dependencies' title='Start of snippet'>anchor</a></sup>
<!-- endSnippet -->
