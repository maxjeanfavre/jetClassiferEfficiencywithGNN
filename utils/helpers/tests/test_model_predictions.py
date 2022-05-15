import uuid
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from utils import Paths
from utils.data.dataframe_format import get_idx_from_event_n_jets
from utils.helpers.model_predictions import ModelPredictions


class TestModelPredictions:
    @pytest.mark.parametrize("with_event_n_jets", [True, False])
    @pytest.mark.parametrize("err_nans", ["none", "some", "all"])
    @pytest.mark.parametrize("res_nans", ["none", "some", "all"])
    @pytest.mark.parametrize("n_jets_per_event", [(1, 10)])
    @pytest.mark.parametrize("n_events", [0, 1, 10, 10 ** 2])
    def test_identity_save_and_load(
        self,
        n_events: int,
        n_jets_per_event: Tuple[int, int],
        res_nans: str,
        err_nans: str,
        with_event_n_jets: bool,
        test_paths: Paths,
    ):
        event_n_jets = np.random.randint(
            n_jets_per_event[0], n_jets_per_event[1] + 1, n_events
        )
        n_jets = np.sum(event_n_jets)

        if res_nans == "none":
            res = np.random.random(size=n_jets)
        elif res_nans == "some":
            res = np.random.random(size=n_jets)
            res[res > 0.9] = np.nan
        elif res_nans == "all":
            res = np.full(shape=n_jets, fill_value=np.nan)
        else:
            raise ValueError(f"Unsupported res_nans: {res_nans}")

        if err_nans == "none":
            err = np.random.random(size=n_jets)
        elif err_nans == "some":
            err = np.random.random(size=n_jets)
            err[err > 0.9] = np.nan
        elif err_nans == "all":
            err = np.full(shape=n_jets, fill_value=np.nan)
        else:
            raise ValueError(f"Unsupported err_nans: {err_nans}")

        if with_event_n_jets:
            idx = get_idx_from_event_n_jets(event_n_jets=event_n_jets)
        else:
            idx = None

        res = pd.Series(res, index=idx)
        err = pd.Series(err, index=idx)

        mp = ModelPredictions(
            res=res,
            err=err,
        )

        dir_path = test_paths.root_path
        filename = f"model_predictions_test_{uuid.uuid4()}.npz"

        mp.save(
            dir_path=dir_path,
            filename=filename,
            event_n_jets=event_n_jets if with_event_n_jets else None,
        )

        mp_loaded = ModelPredictions.load(
            dir_path=dir_path,
            filename=filename,
        )

        pd.testing.assert_series_equal(
            left=mp.res,
            right=mp_loaded.res,
        )

        pd.testing.assert_series_equal(
            left=mp.err,
            right=mp_loaded.err,
        )

        (dir_path / filename).unlink(missing_ok=False)

    def test_errors(self):
        with pytest.raises(ValueError):
            ModelPredictions(
                res=[1],
                err=pd.Series([1]),
            )
        with pytest.raises(ValueError):
            ModelPredictions(
                res=pd.Series([1, 2]),
                err=pd.Series([1]),
            )
        with pytest.raises(ValueError):
            ModelPredictions(
                res=pd.Series(
                    [1, 2], index=pd.MultiIndex.from_arrays(arrays=[[0, 0], [0, 1]])
                ),
                err=pd.Series([1]),
            )
        with pytest.raises(ValueError):
            ModelPredictions(
                res=pd.Series([1]),
                err=[1],
            )
        with pytest.raises(ValueError):
            ModelPredictions(
                res=pd.Series([1]),
                err=pd.Series([1, 2]),
            )
        with pytest.raises(ValueError):
            ModelPredictions(
                res=pd.Series([1, 2]),
                err=pd.Series([1], index=pd.MultiIndex.from_arrays(arrays=[[0], [1]])),
            )
