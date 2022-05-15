from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from utils.data.dataframe_format import get_idx_from_event_n_jets
from utils.data.jet_events_dataset import JetEventsDataset
from utils.data.manipulation.data_filters.jet_multiplicity import JetMultiplicityFilter


class TestJetMultiplicityFilter:
    @pytest.mark.parametrize("n_runs", [10])
    @pytest.mark.parametrize("n_jets_per_event", [(1, 1), (1, 2), (1, 30)])
    @pytest.mark.parametrize("n_events", [1, 10, 10 ** 2, 10 ** 4])
    @pytest.mark.parametrize("mode", ["keep", "remove"])
    def test__manipulate_inplace(
        self,
        mode: str,
        n_events: int,
        n_jets_per_event: Tuple[int, int],
        n_runs: int,
    ):
        for _ in range(n_runs):
            event_n_jets = np.random.randint(
                n_jets_per_event[0], n_jets_per_event[1] + 1, n_events
            )
            n_jets = np.sum(event_n_jets)

            idx = get_idx_from_event_n_jets(event_n_jets=event_n_jets)

            df = pd.DataFrame(index=idx)
            col = "test_col"
            col_data = np.arange(
                n_jets
            )  # values are monotonically increasing and therefore unique
            df[col] = col_data

            for n in range(np.min(event_n_jets), np.max(event_n_jets) + 1 + 1):
                df_jds = df.copy(deep=True)
                jds = JetEventsDataset(df=df_jds)
                jds.manipulate(
                    data_manipulators=(
                        JetMultiplicityFilter(
                            active_modes=("foo",),
                            n=n,
                            mode=mode,
                        ),
                    ),
                    mode="foo",
                )

                if mode == "keep":
                    event_boolean_mask = event_n_jets == n
                elif mode == "remove":
                    event_boolean_mask = event_n_jets != n
                else:
                    raise ValueError(f"Unsupported value for 'mode'. Got: {mode}")

                event_n_jets_expected = event_n_jets[event_boolean_mask]
                np.testing.assert_array_equal(
                    x=jds.event_n_jets,
                    y=event_n_jets_expected,
                )

                df_expected = df.loc[np.where(event_boolean_mask)]
                np.testing.assert_array_equal(
                    x=jds.df[col].to_numpy(),
                    y=df_expected[col].to_numpy(),
                )

    def test_n_0(self):
        with pytest.raises(ValueError):
            JetMultiplicityFilter(
                active_modes=("foo",),
                n=0,
                mode="keep",
            )

    def test_unsupported_mode(self):
        with pytest.raises(ValueError):
            JetMultiplicityFilter(
                active_modes=("foo",),
                n=1,
                mode="foo",
            )
