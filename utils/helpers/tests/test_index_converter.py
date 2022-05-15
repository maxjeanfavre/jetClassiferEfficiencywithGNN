from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from utils.data.dataframe_format import get_idx_from_event_n_jets
from utils.helpers.index_converter import IndexConverter


class TestIndexConverter:
    @pytest.mark.parametrize("n_runs", [20])
    @pytest.mark.parametrize("p", [0.01, 0.3, 0.6, 0.99])
    @pytest.mark.parametrize("n_cols", [2])
    @pytest.mark.parametrize("jets_per_event", [(1, 1), (1, 2), (1, 30)])
    @pytest.mark.parametrize("n_events", [1, 10, 10 ** 2, 10 ** 4])
    def test_get_jet_boolean_mask_from_event_boolean_mask(
        self,
        n_events: int,
        jets_per_event: Tuple[int, int],
        n_cols: int,
        p: float,
        n_runs: int,
    ):
        event_n_jets = np.random.randint(
            jets_per_event[0], jets_per_event[1] + 1, n_events
        )

        idx = get_idx_from_event_n_jets(event_n_jets=event_n_jets)

        n_jets = np.sum(event_n_jets)

        df = pd.DataFrame(
            data=np.random.rand(n_jets, n_cols),
            index=idx,
        )

        for _ in range(n_runs):
            event_boolean_mask = np.random.rand(n_events) > p

            jet_boolean_mask = (
                IndexConverter.get_jet_boolean_mask_from_event_boolean_mask(
                    event_boolean_mask=event_boolean_mask,
                    event_n_jets=event_n_jets,
                )
            )

            df_1 = df.loc[jet_boolean_mask]

            event_index = np.where(event_boolean_mask)[0]
            df_2 = df.loc[event_index]

            pd.testing.assert_frame_equal(
                left=df_1,
                right=df_2,
            )

    def test_get_jet_boolean_mask_from_event_boolean_mask_errors(self):
        with pytest.raises(ValueError):
            IndexConverter.get_jet_boolean_mask_from_event_boolean_mask(
                event_boolean_mask=[True],
                event_n_jets=np.array([1]),
            )
        with pytest.raises(ValueError):
            IndexConverter.get_jet_boolean_mask_from_event_boolean_mask(
                event_boolean_mask=np.array([1]),
                event_n_jets=np.array([1]),
            )
        with pytest.raises(ValueError):
            IndexConverter.get_jet_boolean_mask_from_event_boolean_mask(
                event_boolean_mask=np.array([True]),
                event_n_jets=[1],
            )
        with pytest.raises(ValueError):
            IndexConverter.get_jet_boolean_mask_from_event_boolean_mask(
                event_boolean_mask=np.array([True]),
                event_n_jets=np.array([1.0]),
            )
        with pytest.raises(ValueError):
            IndexConverter.get_jet_boolean_mask_from_event_boolean_mask(
                event_boolean_mask=np.array([[True]]),
                event_n_jets=np.array([1]),
            )
        with pytest.raises(ValueError):
            IndexConverter.get_jet_boolean_mask_from_event_boolean_mask(
                event_boolean_mask=np.array([True]),
                event_n_jets=np.array([[1]]),
            )
        with pytest.raises(ValueError):
            IndexConverter.get_jet_boolean_mask_from_event_boolean_mask(
                event_boolean_mask=np.array([True, False]),
                event_n_jets=np.array([1]),
            )

    @pytest.mark.parametrize("n_runs", [50])
    @pytest.mark.parametrize("n_events", [1, 10, 10 ** 2, 10 ** 4, 10 ** 6])
    def test_get_event_boolean_mask_from_event_index(self, n_events: int, n_runs: int):
        for _ in range(n_runs):
            event_boolean_mask = np.random.rand(n_events) > 0.5

            event_index = np.where(event_boolean_mask)[0]

            event_boolean_mask_reconstructed = (
                IndexConverter.get_event_boolean_mask_from_event_index(
                    event_index=event_index, n_events=n_events
                )
            )

            np.testing.assert_array_equal(
                x=event_boolean_mask_reconstructed,
                y=event_boolean_mask,
            )

    @pytest.mark.parametrize("n_runs", [20])
    @pytest.mark.parametrize("p", [0.01, 0.3, 0.6, 0.99])
    @pytest.mark.parametrize("n_cols", [2])
    @pytest.mark.parametrize("n_rows", [1, 10, 10 ** 2, 10 ** 4])
    def test_get_event_boolean_mask_from_event_index_loc(
        self,
        n_rows: int,
        n_cols: int,
        p: float,
        n_runs: int,
    ):
        df = pd.DataFrame(
            data=np.random.rand(n_rows, n_cols),
        )

        for _ in range(n_runs):
            event_index = np.random.choice(
                a=np.arange(n_rows), size=int(p * n_rows), replace=False
            )

            event_index_sorted = np.sort(event_index)

            event_boolean_mask = IndexConverter.get_event_boolean_mask_from_event_index(
                event_index=event_index_sorted, n_events=n_rows
            )

            df_test = df.loc[event_boolean_mask]

            df_true = df.loc[event_index_sorted]

            pd.testing.assert_frame_equal(
                left=df_test,
                right=df_true,
            )

    def test_get_event_boolean_mask_from_event_index_errors(self):
        with pytest.raises(ValueError):
            IndexConverter.get_event_boolean_mask_from_event_index(
                event_index=[1],
                n_events=1,
            )
        with pytest.raises(ValueError):
            IndexConverter.get_event_boolean_mask_from_event_index(
                event_index=np.array([1.0]),
                n_events=1,
            )
        with pytest.raises(ValueError):
            IndexConverter.get_event_boolean_mask_from_event_index(
                event_index=np.array([[1]]),
                n_events=1,
            )
        with pytest.raises(ValueError):
            IndexConverter.get_event_boolean_mask_from_event_index(
                event_index=np.array([1, 1]),
                n_events=1,
            )
        with pytest.raises(ValueError):
            IndexConverter.get_event_boolean_mask_from_event_index(
                event_index=np.array([1]),
                n_events=1.0,
            )
        with pytest.raises(ValueError):
            IndexConverter.get_event_boolean_mask_from_event_index(
                event_index=np.array([1, 0]),
                n_events=1,
            )
        with pytest.raises(ValueError):
            IndexConverter.get_event_boolean_mask_from_event_index(
                event_index=np.array([0, 1, 2, 3]),
                n_events=2,
            )

    @pytest.mark.parametrize("n_runs", [20])
    @pytest.mark.parametrize("p", [0.01, 0.3, 0.6, 0.99])
    @pytest.mark.parametrize("n_cols", [2])
    @pytest.mark.parametrize("jets_per_event", [(1, 1), (1, 2), (1, 30)])
    @pytest.mark.parametrize("n_events", [1, 10, 10 ** 2, 10 ** 4])
    def test_get_jet_boolean_mask_from_event_boolean_mask_from_event_index_loc(
        self,
        n_events: int,
        jets_per_event: Tuple[int, int],
        n_cols: int,
        p: float,
        n_runs: int,
    ):
        event_n_jets = np.random.randint(
            jets_per_event[0], jets_per_event[1] + 1, n_events
        )

        idx = get_idx_from_event_n_jets(event_n_jets=event_n_jets)

        n_jets = np.sum(event_n_jets)

        df = pd.DataFrame(
            data=np.random.rand(n_jets, n_cols),
            index=idx,
        )

        for _ in range(n_runs):
            event_index = np.random.choice(
                a=np.arange(n_events), size=int(p * n_events), replace=False
            )

            event_index_sorted = np.sort(event_index)

            event_boolean_mask = IndexConverter.get_event_boolean_mask_from_event_index(
                event_index=event_index_sorted, n_events=n_events
            )

            jet_boolean_mask = (
                IndexConverter.get_jet_boolean_mask_from_event_boolean_mask(
                    event_boolean_mask=event_boolean_mask,
                    event_n_jets=event_n_jets,
                )
            )

            df_1 = df.loc[jet_boolean_mask]

            event_index = np.where(event_boolean_mask)[0]
            df_2 = df.loc[event_index]

            pd.testing.assert_frame_equal(
                left=df_1,
                right=df_2,
            )
