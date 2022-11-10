from pathlib import Path
from typing import Tuple, Mapping, Union, Any

import xarray as xr
import zarr
from xarray import Dataset

from eotransform_xarray.storage.storage import Storage

ChunksDefinition = Union[Tuple[int, ...], bool, str]


class StorageUsingZarr(Storage):
    def __init__(self, path: Path):
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> Mapping[str, Any]:
        data = {}
        for sub in self.path.iterdir():
            if sub.is_dir():
                data[sub.stem] = xr.open_zarr(sub)
        return data

    def save(self, data: Mapping[str, Dataset]) -> None:
        for name, ds in data.items():
            ds.to_zarr(self._path / f"{name}.zarr")
