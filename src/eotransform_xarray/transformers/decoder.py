from eotransform_xarray.functional.decode import decode_array
from eotransform_xarray.transformers import TransformerOfDataArray
from xarray import DataArray


class Decoder(TransformerOfDataArray):
    def __call__(self, x: DataArray) -> DataArray:
        return decode_array(x)
