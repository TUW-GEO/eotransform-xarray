from pathlib import Path
from typing import Callable, Any, Dict, Optional, Sequence, Set

import rasterio
import rioxarray
from eotransform.collection_transformation import transform_all_dict_elems

from eotransform.protocol.transformer import PredicatedTransformer
from xarray import DataArray

LEGACY_SCALE_FACTOR_KEYS = {"scale_factor", "Scale_factor"}
SCALE_FACTOR_KEY = "scale_factor"
Parser = Callable[[str], Any]


class PredicatedTagsParser(PredicatedTransformer[Any, Any, Any]):
    def __init__(self, attribute_parsers: Dict[str, Parser]):
        self._attribute_parsers = attribute_parsers

    def is_applicable(self, k: Any) -> bool:
        return k in self._attribute_parsers

    def apply(self, k: Any, x: Any) -> Any:
        return self._attribute_parsers[k](x)


def load_tif(tif: Path, tags_parser: Optional[PredicatedTagsParser] = None, rioxarray_kwargs: Optional[Dict] = None,
             allow_legacy_scaling: Optional[bool] = False):
    rioxarray_kwargs = rioxarray_kwargs or {}

    array = rioxarray.open_rasterio(tif, **rioxarray_kwargs)
    if tags_parser is not None:
        array.attrs = transform_all_dict_elems(array.attrs, tags_parser)

    if allow_legacy_scaling:
        with rasterio.open(tif) as src:
            all_tags = src.tags()
        if _is_loaded_from_legacy_file_format(array, all_tags.keys()):
            array.encoding[SCALE_FACTOR_KEY] = 1 / get_legacy_scale_factor(all_tags)
            array = array * array.encoding[SCALE_FACTOR_KEY]
    return array


def _is_loaded_from_legacy_file_format(array: DataArray, all_array_tags: Set) -> bool:
    return len(LEGACY_SCALE_FACTOR_KEYS.intersection(all_array_tags)) == 1 \
        and (SCALE_FACTOR_KEY not in array.encoding or array.encoding[SCALE_FACTOR_KEY] == 1.0)


def get_legacy_scale_factor(tags: Dict) -> float:
    for key in LEGACY_SCALE_FACTOR_KEYS:
        if key in tags:
            return float(tags[key])
    raise AssertionError(f"Legacy scale factor keys {LEGACY_SCALE_FACTOR_KEYS} not found in {tags}.")
