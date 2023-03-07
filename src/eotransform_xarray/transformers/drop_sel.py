from typing import Dict

from eotransform_xarray.transformers import TransformerOfXArrayData, XArrayData


class DropSel(TransformerOfXArrayData):
    def __init__(self, labels: Dict):
        self._labels = labels

    def __call__(self, x: XArrayData) -> XArrayData:
        return x.drop_sel(self._labels)
