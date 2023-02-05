from xarray import Dataset

from eotransform_xarray.transformers import TransformerOfDataset, TransformerOfDataArray


class ApplyToDataVar(TransformerOfDataset):
    def __init__(self, data_var: str, transformer: TransformerOfDataArray):
        self._data_var = data_var
        self._transformer = transformer

    def __call__(self, x: Dataset) -> Dataset:
        x[self._data_var] = self._transformer(x[self._data_var])
        return x
