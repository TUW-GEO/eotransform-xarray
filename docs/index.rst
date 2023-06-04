.. toctree::
   :hidden:

   Home <self>
   Installation <install>
   API reference <_autosummary/eotransform_xarray>

Welcome to eotransform-xarray's documentation!
==============================================

The eotransform-xarray package provides common transformations on raster data represented as `xarray <https://docs.xarray.dev/en/stable/>`_ data structures, following the Transformer protocol of `eotransform <https://github.com/TUW-GEO/eotransform>`_.
This makes them easy to mix and match, and you can quickly chain processing pipelines, using other `eotransform <https://github.com/TUW-GEO/eotransform) protocols>`_.
Additionally, processing pipelines constructed from these Transformers, can be automatically applied to the `streamed_process <https://eotransform.readthedocs.io/en/latest/_autosummary/eotransform.streamed_process.streamed_process.html#eotransform.streamed_process.streamed_process>`_ function from `eotransform <https://github.com/TUW-GEO/eotransform>`_, to benefit from I/O hiding.

Have a look at the project `README <https://github.com/TUW-GEO/eotransform-xarray/blob/main/README.md>`_ for examples.