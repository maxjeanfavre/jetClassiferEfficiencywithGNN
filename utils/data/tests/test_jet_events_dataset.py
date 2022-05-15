from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from utils.data.dataframe_format import get_idx_from_event_n_jets
from utils.data.jet_events_dataset import JetEventsDataset


class TestJetEventsDatasetSplitData:
    @pytest.mark.parametrize("n_runs", [5])
    @pytest.mark.parametrize("n_seeds", [5])
    @pytest.mark.parametrize("n_jets_per_event", [(1, 1), (1, 2), (1, 30)])
    @pytest.mark.parametrize("n_events", [10, 10 ** 2, 10 ** 4])
    @pytest.mark.parametrize("train_size", [0, 0.1, 0.25, 0.5, 1])
    @pytest.mark.parametrize("mode", ["train_only", "test_only", "both"])
    def test_split_data_random_state(
        self,
        mode: str,
        train_size,
        n_events: int,
        n_jets_per_event: Tuple[int, int],
        n_seeds: int,
        n_runs: int,
    ):
        test_size = 1 - train_size

        for _ in range(n_seeds):
            random_state = np.random.randint(1000)

            event_n_jets = np.random.randint(
                n_jets_per_event[0], n_jets_per_event[1] + 1, n_events
            )

            df = pd.DataFrame(
                index=get_idx_from_event_n_jets(event_n_jets=event_n_jets)
            )

            df["test_col"] = np.arange(
                len(df)
            )  # monotonically increasing -> unique values

            jds = JetEventsDataset(df=df)

            if mode == "train_only" or mode == "test_only":
                if mode == "train_only":
                    return_train = True
                    return_test = False
                else:
                    return_train = False
                    return_test = True

                jds_a = jds.split_data(
                    train_size=train_size,
                    test_size=test_size,
                    return_train=return_train,
                    return_test=return_test,
                    random_state=random_state,
                )

                for _ in range(n_runs):
                    jds_b = jds.split_data(
                        train_size=train_size,
                        test_size=test_size,
                        return_train=return_train,
                        return_test=return_test,
                        random_state=random_state,
                    )

                    assert jds_a == jds_b
            elif mode == "both":
                return_train = True
                return_test = True

                if train_size == 0 or train_size == 1:
                    pass
                else:
                    jds_train, jds_test = jds.split_data(
                        train_size=train_size,
                        test_size=test_size,
                        return_train=return_train,
                        return_test=return_test,
                        random_state=random_state,
                    )

                    for _ in range(n_runs):
                        jds_train_, jds_test_ = jds.split_data(
                            train_size=train_size,
                            test_size=test_size,
                            return_train=return_train,
                            return_test=return_test,
                            random_state=random_state,
                        )

                        assert jds_train == jds_train_
                        assert jds_test == jds_test_
            else:
                raise ValueError(f"Unsupported mode: {mode}")

    def test_split_data_errors(self):
        n_events = 10 ** 2
        n_jets_per_event = (1, 5)

        event_n_jets = np.random.randint(
            n_jets_per_event[0], n_jets_per_event[1] + 1, n_events
        )

        df = pd.DataFrame(index=get_idx_from_event_n_jets(event_n_jets=event_n_jets))

        df["test_col"] = np.arange(len(df))

        jds = JetEventsDataset(df=df)

        with pytest.raises(ValueError):
            jds.split_data(
                train_size=0.5,
                test_size=0.5,
                return_train=False,
                return_test=False,
            )
        with pytest.raises(ValueError):
            jds.split_data(
                train_size=0.6,
                test_size=0.5,
                return_train=False,
                return_test=False,
            )
        with pytest.raises(ValueError):
            jds.split_data(
                train_size=1,
                test_size=0,
                return_train=True,
                return_test=True,
            )
        with pytest.raises(ValueError):
            jds.split_data(
                train_size=0,
                test_size=1,
                return_train=True,
                return_test=True,
            )


class TestJetEventsDatasetGetBootstrapSample:
    @pytest.mark.parametrize("n_runs", [5])
    @pytest.mark.parametrize("n_jets_per_event", [(1, 1), (1, 2), (1, 30)])
    @pytest.mark.parametrize("n_events", [10, 10 ** 2, 10 ** 3])
    @pytest.mark.parametrize("sample_size_frac", [0.1, 0.9, 1, 2, 5])
    def test_get_bootstrap_sample(
        self,
        sample_size_frac,
        n_events: int,
        n_jets_per_event: Tuple[int, int],
        n_runs: int,
    ):
        for _ in range(n_runs):
            event_n_jets = np.random.randint(
                n_jets_per_event[0], n_jets_per_event[1] + 1, n_events
            )

            df = pd.DataFrame(
                index=get_idx_from_event_n_jets(event_n_jets=event_n_jets)
            )

            df["test_col_a"] = np.arange(
                len(df)
            )  # monotonically increasing -> unique values
            df["test_col_b"] = np.random.random(size=len(df))

            jds = JetEventsDataset(df=df)

            sample_size = int(sample_size_frac * jds.n_events)

            jds_bootstrap, idxs = jds.get_bootstrap_sample(
                sample_size=sample_size,
                return_bootstrap_idxs=True,
            )

            bootstrap_selection = np.concatenate(idxs)

            df = pd.concat(
                objs=[jds.df.loc[x] for x in bootstrap_selection],
                keys=[i for i in range(len(bootstrap_selection))],
                sort=False,
            )
            df.index.names = jds_bootstrap.df.index.names

            pd.testing.assert_frame_equal(
                left=jds_bootstrap.df,
                right=df,
            )

    def test_errors(self):
        n_events = 10 ** 2
        n_jets_per_event = (1, 5)

        event_n_jets = np.random.randint(
            n_jets_per_event[0], n_jets_per_event[1] + 1, n_events
        )

        df = pd.DataFrame(index=get_idx_from_event_n_jets(event_n_jets=event_n_jets))

        df["test_col"] = np.arange(len(df))

        jds = JetEventsDataset(df=df)

        with pytest.raises(ValueError):
            jds.get_bootstrap_sample(sample_size=0)
