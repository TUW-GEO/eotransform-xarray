from pathlib import Path
from typing import Any, Dict, Callable, Optional, Sequence

import rasterio
import rioxarray
import xarray as xr
from eotransform.collection_transformation import transform_all_dict_elems
from eotransform.protocol.transformer import PredicatedTransformer, Transformer
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


class FileDataFrameToDataArray(Transformer[DataFrame, DataArray]):
    def __init__(self, registered_attribute_parsers: Optional[Dict[str, Parser]] = None):
        self._tags_parser = PredicatedTagsParser(registered_attribute_parsers or {})

    def __call__(self, x: DataFrame) -> DataArray:
        index_name = x.index.name
        arrays = [_to_data_array(row, index, index_name, self._tags_parser) for index, row in x.iterrows()]
        return xr.concat(arrays, dim=index_name, combine_attrs=_concat_attrs_with_key(CONCATED_ATTRS_KEY))


def _to_data_array(row: Series, index: Any, index_name: str, tags_parser: PredicatedTagsParser) -> DataArray:
    if 'filepath' in row:
        return _read_geo_tiff(row['filepath'], index, index_name, tags_parser)
    elif 'filepaths' in row:
        return _read_multi_band_geo_tiffs(row['filepaths'], index, index_name, tags_parser)
    else:
        raise NotImplementedError(f'Reading geo tiffs from pandas series {row} not implemented.')


def _read_multi_band_geo_tiffs(tiffs: Sequence[Path], index: Any, index_name: str,
                               tags_parser: PredicatedTagsParser) -> DataArray:
    arrays = [_read_array_from_tif(t, tags_parser) for t in tiffs]
    array = xr.concat(arrays, dim='band', combine_attrs=_concat_attrs_with_key(BAND_ATTRS_KEY))
    return array.expand_dims(index_name).assign_coords({'band': [i for i in range(len(arrays))], index_name: [index]})


def _read_array_from_tif(tif, tags_parser):
    with rasterio.open(tif) as rds:
        array = rioxarray.open_rasterio(rds).chunk()
        tags = transform_all_dict_elems(rds.tags(), tags_parser)
        array.attrs = {**array.attrs, **tags}
    return array


def _read_geo_tiff(tif: Path, index: Any, index_name: str, tags_parser: PredicatedTagsParser) -> DataArray:
    array = _read_array_from_tif(tif, tags_parser)
    return array.expand_dims(index_name).assign_coords({index_name: [index]})


def _concat_attrs_with_key(key: str):
    return lambda attrs, context: {key: attrs}
