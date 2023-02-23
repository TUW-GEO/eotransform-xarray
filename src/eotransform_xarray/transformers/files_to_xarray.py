from typing import Dict, Optional

from eotransform.protocol.transformer import Transformer
from pandas import DataFrame
from xarray import DataArray

from eotransform_xarray.functional.load_file_dataframe_to_array import load_file_dataframe_to_array
from eotransform_xarray.functional.load_tif import Parser


class FileDataFrameToDataArray(Transformer[DataFrame, DataArray]):
    def __init__(self, registered_attribute_parsers: Optional[Dict[str, Parser]] = None,
                 rioxarray_kwargs: Optional[Dict] = None):
        self._registered_attribute_parsers = registered_attribute_parsers
        self._rioxarray_kwargs = rioxarray_kwargs or {}

    def __call__(self, x: DataFrame) -> DataArray:
        return load_file_dataframe_to_array(x, self._registered_attribute_parsers, self._rioxarray_kwargs)
