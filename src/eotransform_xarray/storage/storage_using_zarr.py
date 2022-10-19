import zarr

from pathlib import Path
from typing import Optional, Tuple, Mapping, Union, Any
from numpy.typing import ArrayLike

from eotransform_xarray.storage.storage import Storage

ChunksDefinition = Union[Tuple[int, ...], bool, str]


class StorageUsingZarr(Storage):
    def __init__(self, zarr_path: Path):
        self._path = zarr_path

    def exists(self) -> bool:
        return self._path.exists()

    def load(self) -> Mapping[str, Any]:
        return zarr.open_group(self._path, mode='r')

    def save(self, data: Mapping[str, ArrayLike]) -> None:
        group = zarr.open_group(self._path, mode='w')
        for name, values in data.items():
            group.array(name, values)
