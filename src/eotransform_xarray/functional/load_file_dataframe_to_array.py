from pathlib import Path
from typing import Callable, Any, Dict, Optional, Sequence

import numpy as np
import rasterio
import rioxarray
import xarray as xr
from eotransform.collection_transformation import transform_all_dict_elems
from eotransform.protocol.transformer import PredicatedTransformer
from pandas import DataFrame, Series

from xarray import DataArray

CONCATED_ATTRS_KEY = 'concated_attrs'
BAND_ATTRS_KEY = 'band_attrs'

Parser = Callable[[str], Any]


class PredicatedTagsParser(PredicatedTransformer[Any, Any, Any]):
    def __init__(self, attribute_parsers: Dict[str, Parser]):
        self._attribute_parsers = attribute_parsers

    def is_applicable(self, k: Any) -> bool:
        return k in self._attribute_parsers

    def apply(self, k: Any, x: Any) -> Any:
        return self._attribute_parsers[k](x)


def load_file_dataframe_to_array(x: DataFrame,
                                 registered_attribute_parsers: Optional[Dict[str, Parser]] = None,
                                 open_rasterio_kwargs: Optional[Dict] = None,
                                 rasterio_open_kwargs: Optional[Dict] = None) -> DataArray:
    tags_parser = PredicatedTagsParser(registered_attribute_parsers or {})
    open_rasterio_kwargs = open_rasterio_kwargs or {}
    rasterio_open_kwargs = rasterio_open_kwargs or {}
    index_name = x.index.name
    arrays = [_to_data_array(row, index, index_name, tags_parser, rasterio_open_kwargs, open_rasterio_kwargs)
              for index, row in x.iterrows()]
    return xr.concat(arrays, dim=index_name, combine_attrs=_concat_attrs_with_key(CONCATED_ATTRS_KEY))


def _to_data_array(row: Series, index: Any, index_name: str, tags_parser: PredicatedTagsParser,
                   rasterio_open_kwargs: Dict, open_rasterio_kwargs: Dict) -> DataArray:
    if 'filepath' in row:
        return _read_geo_tiff(row['filepath'], index, index_name, tags_parser, rasterio_open_kwargs,
                              open_rasterio_kwargs)
    elif 'filepaths' in row:
        return _read_multi_band_geo_tiffs(row['filepaths'], index, index_name, tags_parser, rasterio_open_kwargs,
                                          open_rasterio_kwargs)
    else:
        raise NotImplementedError(f'Reading geo tiffs from pandas series {row} not implemented.')


def _read_geo_tiff(tif: Path, index: Any, index_name: str, tags_parser: PredicatedTagsParser,
                   rasterio_open_kwargs: Dict, open_rasterio_kwargs: Dict) -> DataArray:
    array = _read_array_from_tif(tif, tags_parser, rasterio_open_kwargs, open_rasterio_kwargs)
    return array.expand_dims(index_name).assign_coords({index_name: (index_name, [index]),
                                                        "filepath": (index_name, [tif])})


def _read_multi_band_geo_tiffs(tiffs: Sequence[Path], index: Any, index_name: str,
                               tags_parser: PredicatedTagsParser, rasterio_open_kwargs: Dict,
                               open_rasterio_kwargs: Dict) -> DataArray:
    arrays = [_read_array_from_tif(t, tags_parser, rasterio_open_kwargs, open_rasterio_kwargs) for t in tiffs]
    array = xr.concat(arrays, dim='band', combine_attrs=_concat_attrs_with_key(BAND_ATTRS_KEY))
    tiff_array = np.empty((1,), dtype=np.object)
    tiff_array[0] = tiffs
    return array.expand_dims(index_name).assign_coords(
        {'band': [i for i in range(len(arrays))], index_name: [index], "filepaths": (index_name, tiff_array)})


def _read_array_from_tif(tif, tags_parser, rasterio_open_kwargs, open_rasterio_kwargs):
    with rasterio.open(tif, **rasterio_open_kwargs) as rds:
        array = rioxarray.open_rasterio(rds, **open_rasterio_kwargs)
        tags = transform_all_dict_elems(rds.tags(), tags_parser)
        array.attrs['tags'] = tags
    return array


def _concat_attrs_with_key(key: str):
    return lambda attrs, context: {key: attrs}
