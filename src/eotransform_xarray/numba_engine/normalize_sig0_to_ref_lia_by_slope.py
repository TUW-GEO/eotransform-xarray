from numpy._typing import NDArray

try:
    from numba import njit, prange
except ImportError:
    print("sig0 normalization with numba engine requires numba.\npip install numba")
    raise


@njit(parallel=True)
def normalize_numba(sig0: NDArray, slope: NDArray, lia: NDArray, ref_lia: float, out: NDArray) -> None:
    for y in prange(sig0.shape[-2]):
        for x in prange(sig0.shape[-1]):
            out[0, y, x] = sig0[0, y, x] - slope[0, y, x] * (lia[0, y, x] - ref_lia)
