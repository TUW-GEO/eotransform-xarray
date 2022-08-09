from eotransform.protocol.transformer import Transformer
from pandas import DataFrame


class GroupToBands(Transformer[DataFrame, DataFrame]):
    class NumInputFilesMismatchError(AssertionError):
        ...

    def __init__(self, num_bands: int):
        self._num_bands = num_bands

    def __call__(self, x: DataFrame) -> DataFrame:
        if len(x) % self._num_bands != 0:
            msg = f"{len(x)} input files is not a multiple of requested bands {self._num_bands}"
            raise GroupToBands.NumInputFilesMismatchError(msg)

        x['band_id'] = [bi for bi in range(0, len(x), self._num_bands) for _ in range(self._num_bands)]
        per_band_files = x.groupby('band_id')['filepath'].apply(list)
        x = x.join(per_band_files, on='band_id', how='right', rsuffix='s')
        x.drop_duplicates(['band_id'], keep='first', inplace=True)
        x.drop('filepath', axis=1, inplace=True)
        x.drop('band_id', axis=1, inplace=True)
        return x
