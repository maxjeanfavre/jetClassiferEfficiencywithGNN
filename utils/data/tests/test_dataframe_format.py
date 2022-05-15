from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from utils.data.dataframe_format import (
    check_zero_level_values_for_repeated_values,
    get_first_level_values_from_event_n_jets_njit,
    get_idx_from_event_n_jets,
    get_zero_level_values_from_event_n_jets,
    reconstruct_event_n_jets_from_groupby_zeroth_level_values,
)


class TestGetIdxFromEventNJets:
    @pytest.mark.parametrize("n_jets_per_event", [(1, 1), (1, 2), (1, 30)])
    @pytest.mark.parametrize("n_events", [0, 1, 10, 10 ** 2, 10 ** 4])
    def test_get_idx_from_event_n_jets(
        self, n_events: int, n_jets_per_event: Tuple[int, int]
    ):
        event_n_jets = np.random.randint(
            n_jets_per_event[0], n_jets_per_event[1] + 1, n_events
        )

        idx = get_idx_from_event_n_jets(event_n_jets=event_n_jets)

        assert isinstance(idx, pd.MultiIndex)

    @pytest.mark.parametrize(
        "event_n_jets,expected_exception",
        [
            ([1], pytest.raises(ValueError)),
            (np.array([[1]]), pytest.raises(ValueError)),
            (np.array([0, 1]), pytest.raises(ValueError)),
        ],
    )
    def test_errors(self, event_n_jets, expected_exception):
        with expected_exception:
            get_idx_from_event_n_jets(event_n_jets=event_n_jets)


class TestGetZeroLevelValuesFromEventNJets:
    @pytest.mark.parametrize("n_runs", [10])
    @pytest.mark.parametrize("n_jets_per_event", [(1, 1), (1, 2), (1, 30)])
    @pytest.mark.parametrize("n_events", [0, 1, 10, 10 ** 2, 10 ** 4])
    def test_get_zero_level_values_from_event_n_jets(
        self, n_events: int, n_jets_per_event: Tuple[int, int], n_runs: int
    ):
        for _ in range(n_runs):
            event_n_jets = np.random.randint(
                n_jets_per_event[0], n_jets_per_event[1] + 1, n_events
            )

            zero_level_values = get_zero_level_values_from_event_n_jets(
                event_n_jets=event_n_jets
            )

            zero_level_values_manual = np.array(
                [i for i, n_jets in enumerate(event_n_jets) for _ in range(n_jets)]
            )

            np.testing.assert_array_equal(
                x=zero_level_values,
                y=zero_level_values_manual,
            )


class TestGetFirstLevelValuesFromEventNJetsNjit:
    @pytest.mark.parametrize("n_runs", [10])
    @pytest.mark.parametrize("n_jets_per_event", [(1, 1), (1, 2), (1, 30)])
    @pytest.mark.parametrize("n_events", [0, 1, 10, 10 ** 2, 10 ** 4])
    def test_get_first_level_values_from_event_n_jets_njit(
        self, n_events: int, n_jets_per_event: Tuple[int, int], n_runs: int
    ):
        for _ in range(n_runs):
            event_n_jets = np.random.randint(
                n_jets_per_event[0], n_jets_per_event[1] + 1, n_events
            )

            with_njit = get_first_level_values_from_event_n_jets_njit(
                event_n_jets=event_n_jets
            )

            without_njit_1 = np.array(
                [i for n_jets in event_n_jets for i in range(n_jets)]
            )

            np.testing.assert_array_equal(
                x=with_njit,
                y=without_njit_1,
            )

            if n_events != 0:
                without_njit_2 = np.concatenate(
                    tuple(np.arange(n_jets) for n_jets in event_n_jets)
                )

                np.testing.assert_array_equal(
                    x=with_njit,
                    y=without_njit_2,
                )


class TestReconstructEventNJetsFromGroupbyZerothLevelValues:
    @pytest.mark.parametrize("n_runs", [10])
    @pytest.mark.parametrize("n_jets_per_event", [(1, 1), (1, 2), (1, 30)])
    @pytest.mark.parametrize("n_events", [0, 1, 10, 10 ** 2, 10 ** 4])
    def test_reconstruct_event_n_jets_from_groupby_zeroth_level_values(
        self, n_events: int, n_jets_per_event: Tuple[int, int], n_runs: int
    ):
        for _ in range(n_runs):
            event_n_jets = np.random.randint(
                n_jets_per_event[0], n_jets_per_event[1] + 1, n_events
            )

            zero_level_values = get_zero_level_values_from_event_n_jets(
                event_n_jets=event_n_jets
            )

            df = pd.DataFrame(
                index=pd.MultiIndex.from_arrays(arrays=[zero_level_values])
            )

            reconstructed_event_n_jets = (
                reconstruct_event_n_jets_from_groupby_zeroth_level_values(df=df)
            )

            np.testing.assert_array_equal(
                x=reconstructed_event_n_jets,
                y=event_n_jets,
            )

    def test_errors(self):
        with pytest.raises(ValueError):
            df = pd.DataFrame(
                index=pd.MultiIndex.from_arrays(
                    arrays=[
                        np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 4, 4, 4])
                    ]
                )
            )
            reconstruct_event_n_jets_from_groupby_zeroth_level_values(df=df)


class TestCheckZeroLevelValuesForRepeatedValues:
    def test_check_zero_level_values_for_repeated_values_error(self):
        zero_level_values = np.array(
            [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 4, 4, 4]
        )

        with pytest.raises(ValueError):
            check_zero_level_values_for_repeated_values(
                zero_level_values=zero_level_values
            )

    @pytest.mark.parametrize("n_jets_per_event", [(1, 1), (1, 2), (1, 30)])
    @pytest.mark.parametrize("n_events", [0, 1, 10, 10 ** 2, 10 ** 4])
    def test_check_zero_level_values_for_repeated_values_no_error(
        self, n_events: int, n_jets_per_event: Tuple[int, int]
    ):
        event_n_jets = np.random.randint(
            n_jets_per_event[0], n_jets_per_event[1] + 1, n_events
        )

        zero_level_values = get_zero_level_values_from_event_n_jets(
            event_n_jets=event_n_jets
        )

        check_zero_level_values_for_repeated_values(zero_level_values=zero_level_values)
