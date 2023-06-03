from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import xarray as xr
from pandas import DataFrame, Series

from xarray import DataArray

from eotransform_xarray.functional.load_tif import Parser, PredicatedTagsParser, load_tif

CONCATED_ATTRS_KEY = 'concated_attrs'
BAND_ATTRS_KEY = 'band_attrs'
FILEPATH_COORD = 'filepath'


def load_file_dataframe_to_array(x: DataFrame,
                                 registered_attribute_parsers: Optional[Dict[str, Parser]] = None,
                                 rioxarray_kwargs: Optional[Dict] = None,
                                 allow_legacy_scaling: Optional[bool] = False) -> DataArray:
    tags_parser = PredicatedTagsParser(registered_attribute_parsers or {})
    rioxarray_kwargs = rioxarray_kwargs or {}
    index_name = x.index.name
    arrays = [_to_data_array(row, index, index_name, tags_parser, rioxarray_kwargs, allow_legacy_scaling)
              for index, row in x.iterrows()]
    return xr.concat(arrays, dim=index_name, combine_attrs=_concat_attrs_with_key(CONCATED_ATTRS_KEY))


def _to_data_array(row: Series, index: Any, index_name: str, tags_parser: PredicatedTagsParser,
                   rioxarray_kwargs: Dict, allow_legacy_scaling: bool) -> DataArray:
    if 'filepath' in row:
        return _read_geo_tiff(row['filepath'], index, index_name, tags_parser, rioxarray_kwargs, allow_legacy_scaling)
    elif 'filepaths' in row:
        return _read_multi_band_geo_tiffs(row['filepaths'], index, index_name, tags_parser, rioxarray_kwargs,
                                          allow_legacy_scaling)
    else:
        raise NotImplementedError(f'Reading geo tiffs from pandas series {row} not implemented.')


def _read_geo_tiff(tif: Path, index: Any, index_name: str, tags_parser: PredicatedTagsParser,
                   rioxarray_kwargs: Dict, allow_legacy_scaling: bool) -> DataArray:
    array = load_tif(tif, tags_parser, rioxarray_kwargs, allow_legacy_scaling)
    return array.expand_dims(index_name).assign_coords({index_name: (index_name, [index]),
                                                        FILEPATH_COORD: (index_name, [tif])})


def _read_multi_band_geo_tiffs(tiffs: Sequence[Path], index: Any, index_name: str,
                               tags_parser: PredicatedTagsParser, rioxarray_kwargs: Dict,
                               allow_legacy_scaling: bool) -> DataArray:
    arrays = [load_tif(t, tags_parser, rioxarray_kwargs, allow_legacy_scaling)
              for t in tiffs]
    array = xr.concat(arrays, dim='band', combine_attrs=_concat_attrs_with_key(BAND_ATTRS_KEY))
    tiff_array = np.empty((1,), dtype=object)
    tiff_array[0] = tiffs
    return array.expand_dims(index_name).assign_coords(
        {'band': [i for i in range(len(arrays))], index_name: [index], "filepaths": (index_name, tiff_array)})


def _concat_attrs_with_key(key: str):
    return lambda attrs, context: {key: attrs}
