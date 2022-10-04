from pathlib import Path
from typing import Any, Dict, Callable, Optional, Sequence

import numpy as np
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
    def __init__(self, registered_attribute_parsers: Optional[Dict[str, Parser]] = None,
                 open_rasterio_kwargs: Optional[Dict] = None,
                 rasterio_open_kwargs: Optional[Dict] = None, ):
        self._tags_parser = PredicatedTagsParser(registered_attribute_parsers or {})
        self._open_rasterio_kwargs = open_rasterio_kwargs or {}
        self._rasterio_open_kwargs = rasterio_open_kwargs or {}

    def __call__(self, x: DataFrame) -> DataArray:
        index_name = x.index.name
        arrays = [self._to_data_array(row, index, index_name, self._tags_parser) for index, row in x.iterrows()]
        return xr.concat(arrays, dim=index_name, combine_attrs=_concat_attrs_with_key(CONCATED_ATTRS_KEY))

    def _to_data_array(self, row: Series, index: Any, index_name: str, tags_parser: PredicatedTagsParser) -> DataArray:
        if 'filepath' in row:
            return self._read_geo_tiff(row['filepath'], index, index_name, tags_parser)
        elif 'filepaths' in row:
            return self._read_multi_band_geo_tiffs(row['filepaths'], index, index_name, tags_parser)
        else:
            raise NotImplementedError(f'Reading geo tiffs from pandas series {row} not implemented.')

    def _read_geo_tiff(self, tif: Path, index: Any, index_name: str, tags_parser: PredicatedTagsParser) -> DataArray:
        array = self._read_array_from_tif(tif, tags_parser)
        return array.expand_dims(index_name).assign_coords({index_name: (index_name, [index]),
                                                            "filepath": (index_name, [tif])})

    def _read_multi_band_geo_tiffs(self, tiffs: Sequence[Path], index: Any, index_name: str,
                                   tags_parser: PredicatedTagsParser) -> DataArray:
        arrays = [self._read_array_from_tif(t, tags_parser) for t in tiffs]
        array = xr.concat(arrays, dim='band', combine_attrs=_concat_attrs_with_key(BAND_ATTRS_KEY))
        tiff_array = np.empty((1,), dtype=np.object)
        tiff_array[0] = tiffs
        return array.expand_dims(index_name).assign_coords(
            {'band': [i for i in range(len(arrays))], index_name: [index], "filepaths": (index_name, tiff_array)})

    def _read_array_from_tif(self, tif, tags_parser):
        with rasterio.open(tif, **self._rasterio_open_kwargs) as rds:
            array = rioxarray.open_rasterio(rds, **self._open_rasterio_kwargs)
            tags = transform_all_dict_elems(rds.tags(), tags_parser)
            array.attrs['tags'] = tags
        return array


def _concat_attrs_with_key(key: str):
    return lambda attrs, context: {key: attrs}
