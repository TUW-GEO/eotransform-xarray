from dataclasses import dataclass
from typing import Dict, MutableMapping, Iterator

import numpy as np
from eotransform.protocol.transformer import Transformer
from xarray import DataArray, Dataset


@dataclass
class NDArrayWithAttributes:
    values: np.ndarray
    attrs: Dict


@dataclass
class NDArrayMapping(MutableMapping):
    variables: Dict[str, NDArrayWithAttributes]
    attrs: Dict

    def __setitem__(self, k: str, v: NDArrayWithAttributes) -> None:
        self.variables.__setitem__(k, v)

    def __getitem__(self, k: str) -> NDArrayWithAttributes:
        return self.variables.__getitem__(k)

    def __delitem__(self, v: str) -> None:
        self.variables.__delitem__(v)

    def __len__(self) -> int:
        return self.variables.__len__()

    def __iter__(self) -> Iterator[str]:
        return self.variables.__iter__()


class DataArrayToNumpy(Transformer[DataArray, NDArrayWithAttributes]):
    def __call__(self, x: DataArray) -> NDArrayWithAttributes:
        return NDArrayWithAttributes(x.to_numpy(), x.attrs)


class DatasetToNumpy(Transformer[Dataset, NDArrayMapping]):
    def __init__(self):
        self._array_to_np = DataArrayToNumpy()

    def __call__(self, x: Dataset) -> NDArrayMapping:
        return NDArrayMapping({k: self._array_to_np(v) for k, v in x.items()}, x.attrs)
