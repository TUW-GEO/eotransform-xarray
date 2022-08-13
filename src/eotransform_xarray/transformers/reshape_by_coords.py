from eotransform_xarray.transformers import TransformerOfXArrayData, XArrayData


class ReshapeByCoords(TransformerOfXArrayData):
    def __init__(self, reshape_dim, coords):
        self._reshape_dim = reshape_dim
        self._coords = coords

    def __call__(self, x: XArrayData) -> XArrayData:
        multi_coord = ''.join(self._coords)
        return x.swap_dims({self._reshape_dim: self._coords[0]}) \
            .set_index({multi_coord: self._coords}) \
            .unstack(multi_coord)
