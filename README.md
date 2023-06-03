![Coverage badge](https://raw.githubusercontent.com/TUW-GEO/eotransform-xarray/python-coverage-comment-action-data/badge.svg)
# eotransform-xarray

## What can I use eotransform-xarray for?

The eotransform-xarray package provides common transformations on raster data represented as [xarray](https://docs.xarray.dev/en/stable/) data structures, following the Transformer protocol of [eotransform](https://github.com/TUW-GEO/eotransform).
This makes them easy to mix and match, and you can quickly chain processing pipelines, using other [eotransform](https://github.com/TUW-GEO/eotransform) protocols.
Also benefit from streaming function

## Getting Started
### Installation
```bash
pip install eotransform-xarray
```

### Example: streamed processing pipeline
In the following example swath data is resampled, masked and written out as a GeoTIFF stack.

snippet: streamed_resample_and_mask

Note, that this example uses [eotransform](https://github.com/TUW-GEO/eotransform)'s `stream` function to hide the I/O operations, using the compute resources more effectively.

### Dependencies:
eotransform-xarray requires Python 3.8 and has these dependencies:

snippet: dependencies