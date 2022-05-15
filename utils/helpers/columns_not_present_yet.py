from typing import Tuple

import pandas as pd


def check_columns_not_present_yet(df: pd.DataFrame, cols: Tuple[str, ...]):
    cols_already_present = set()

    for col in cols:
        if col in df.columns:
            cols_already_present.add(col)

    if cols_already_present:
        raise ValueError(
            f"The columns {cols_already_present} are already present in the data"
        )
