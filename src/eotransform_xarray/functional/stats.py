from typing import Optional

import numpy as np
import scipy
from numpy._typing import NDArray


def calc_person_r_element_wise(x: NDArray, y: NDArray, mask: NDArray, limit: int, out: NDArray) -> None:
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            m = ~mask[i, j]
            xt = np.extract(m, x[i, j])
            yt = np.extract(m, y[i, j])
            if len(xt) > limit:
                out[i, j, 0] = scipy.stats.pearsonr(xt, yt).statistic
                conf = scipy.stats.pearsonr(xt, yt).confidence_interval()
                out[i, j, 1] = conf.low
                out[i, j, 2] = conf.high


def pearsonr(x: NDArray, y: NDArray, min_obs: Optional[int] = 2) -> NDArray:
    nan_mask = np.isnan(x) | np.isnan(y)
    x[nan_mask] = np.nan
    y[nan_mask] = np.nan

    r = np.full(x.shape[:2] + (3,), np.nan)
    calc_person_r_element_wise(x, y, nan_mask, min_obs, r)
    return r
