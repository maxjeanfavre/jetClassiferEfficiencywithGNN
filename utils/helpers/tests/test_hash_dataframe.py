import numpy as np
import pandas as pd
import pytest

from utils.helpers.hash_dataframe import hash_df


class TestHashDf:
    @pytest.mark.parametrize("n_columns", [1, 2, 10])
    @pytest.mark.parametrize("n_rows", [1, 10, 10 ** 3])
    def test_hash_df(self, n_rows, n_columns):
        df = pd.DataFrame(np.random.random(size=(n_rows, n_columns)))

        hash = hash_df(df=df)

        n = 10 ** 2

        for _ in range(n):
            hash_ = hash_df(df=df)
            assert hash == hash_

        for _ in range(n):
            df_ = df.copy(deep=True)
            i = np.random.randint(low=0, high=n_rows)
            j = np.random.randint(low=0, high=n_columns)
            df_.iloc[i, j] = 2
            hash_ = hash_df(df=df_)
            assert hash != hash_

        for _ in range(n):
            df_ = df.copy(deep=True)
            i = np.random.randint(low=0, high=n_columns)
            df_ = df_.rename(columns={i: n_columns + 1})
            hash_ = hash_df(df=df_)
            assert hash != hash_
