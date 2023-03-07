from eotransform_xarray.transformers import TransformerOfXArrayData, XArrayData


class Squeeze(TransformerOfXArrayData):
    def __init__(self, dim=None, drop=False, axis=None):
        self._dim = dim
        self._drop = drop
        self._axis = axis

    def __call__(self, x: XArrayData) -> XArrayData:
        return x.squeeze(self._dim, self._drop, self._axis)
