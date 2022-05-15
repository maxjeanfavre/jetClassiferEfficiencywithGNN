import hashlib

import pandas as pd


def hash_df(df: pd.DataFrame):
    # rudimentary implementation
    content_hash = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()
    columns_hash = hashlib.sha256(df.columns.values).hexdigest()

    hash_value = hashlib.sha256((content_hash + columns_hash).encode()).hexdigest()

    return hash_value
