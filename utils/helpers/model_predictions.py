from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import pandas as pd

from utils.data.dataframe_format import get_idx_from_event_n_jets


class ModelPredictions:
    def __init__(self, res: pd.Series, err: pd.Series) -> None:
        self._res = None
        self._err = None

        self.res = res
        self.err = err

    @property
    def res(self) -> pd.Series:
        return self._res

    @res.setter
    def res(self, value: pd.Series):
        if not isinstance(value, pd.Series):
            raise ValueError("Argument must be a pd.Series")
        if self.err is not None:
            if len(value) != len(self.err):
                raise ValueError("Must be same length as 'err'")
            if (value.index != self.err.index).any():
                raise ValueError("Index is not equal to the index of existing 'err'")

        self._res = value

    @property
    def err(self) -> pd.Series:
        return self._err

    @err.setter
    def err(self, value: pd.Series):
        if not isinstance(value, pd.Series):
            raise ValueError("Argument must be a pd.Series")
        if self.res is not None:
            if len(value) != len(self.res):
                raise ValueError("Must be same length as 'res'")
            if (value.index != self.res.index).any():
                raise ValueError("Index is not equal to the index of existing 'res'")

        self._err = value

    def save(
        self, dir_path: pathlib.Path, filename: str, event_n_jets: Optional[np.ndarray]
    ) -> None:
        d = {}

        if event_n_jets is not None:
            d["event_n_jets"] = event_n_jets

        d["res"] = self.res.to_numpy()

        if not self.err.isnull().all():
            d["err"] = self.err.to_numpy()

        np.savez_compressed(
            file=dir_path / filename,
            **d,
        )

    @classmethod
    def load(cls, dir_path: pathlib.Path, filename: str) -> ModelPredictions:
        loaded = np.load(file=str((dir_path / filename).absolute()))

        res = loaded["res"]

        if "err" in loaded:
            err = loaded["err"]
        else:
            err = np.full_like(a=res, fill_value=np.nan)

        if "event_n_jets" in loaded:
            event_n_jets = loaded["event_n_jets"]
            idx = get_idx_from_event_n_jets(event_n_jets=event_n_jets)
        else:
            idx = None

        res = pd.Series(res, index=idx)
        err = pd.Series(err, index=idx)

        inst = cls(res=res, err=err)

        return inst
