from typing import Tuple

import numpy as np
from affine import Affine
from numpy.typing import NDArray


def pixel_centers_for_geotransform_extent(transform: Affine, columns: int, rows: int) -> Tuple[NDArray, NDArray]:
    x_res = transform.a
    y_res = transform.e
    x_off = transform.xoff + x_res / 2.0
    y_off = transform.yoff + y_res / 2.0
    return np.arange(y_off, y_off + rows * y_res, y_res), np.arange(x_off, x_off + columns * x_res, x_res)
