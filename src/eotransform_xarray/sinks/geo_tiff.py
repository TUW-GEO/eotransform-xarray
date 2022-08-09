from pathlib import Path
from typing import Union, Callable

from xarray import DataArray

from eotransform_xarray.sinks import DataArraySink


class SinkToGeoTiff(DataArraySink):
    def __init__(self, out_dir: Union[Path, str], namer: Callable[[int, DataArray], str]):
        self._out_dir = Path(out_dir)
        self._namer = namer
        self._n = 0

    def __call__(self, x: DataArray) -> None:
        file_name = self._namer(self._n, x)
        x.rio.to_raster(self._out_dir / file_name)
        self._n += 1
