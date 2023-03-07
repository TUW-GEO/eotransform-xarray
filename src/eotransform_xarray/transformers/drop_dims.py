from eotransform_xarray.transformers import TransformerOfXArrayData, XArrayData


class DropDims(TransformerOfXArrayData):
    def __init__(self, *dims: str):
        self._dims = dims

    def __call__(self, x: XArrayData) -> XArrayData:
        return x.drop_dims(self._dims)
