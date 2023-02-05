from typing import Dict, Optional

from eotransform.protocol.transformer import Transformer
from pandas import DataFrame
from xarray import DataArray

from eotransform_xarray.functional.load_file_dataframe_to_array import Parser, load_file_dataframe_to_array


class FileDataFrameToDataArray(Transformer[DataFrame, DataArray]):
    def __init__(self, registered_attribute_parsers: Optional[Dict[str, Parser]] = None,
                 open_rasterio_kwargs: Optional[Dict] = None,
                 rasterio_open_kwargs: Optional[Dict] = None):
        self._registered_attribute_parsers = registered_attribute_parsers
        self._open_rasterio_kwargs = open_rasterio_kwargs or {}
        self._rasterio_open_kwargs = rasterio_open_kwargs or {}

    def __call__(self, x: DataFrame) -> DataArray:
        return load_file_dataframe_to_array(x, self._registered_attribute_parsers, self._open_rasterio_kwargs,
                                            self._rasterio_open_kwargs)
