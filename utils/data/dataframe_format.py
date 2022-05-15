import numpy as np
import pandas as pd

from numba import njit


def get_idx_from_event_n_jets(event_n_jets: np.ndarray):
    if not isinstance(event_n_jets, np.ndarray):
        raise ValueError(
            f"'event_n_jets' has to be np.ndarray. Got type: {type(event_n_jets)}"
        )
    if event_n_jets.ndim != 1:
        raise ValueError(
            "'event_n_jets' has to be 1D np.ndarray. "
            f"Got shape: {event_n_jets.shape}"
        )

    if 0 in event_n_jets:
        raise ValueError("event_n_jets can't contain a 0")
    idx = pd.MultiIndex.from_arrays(
        arrays=[
            # 0th level - njit no real advantage
            get_zero_level_values_from_event_n_jets(event_n_jets=event_n_jets),
            # 1st level
            # np.array([i for n_jets in event_n_jets for i in range(n_jets)]),
            get_first_level_values_from_event_n_jets_njit(event_n_jets=event_n_jets),
        ],
        names=("entry", "subentry"),
    )

    assert all(dtype == np.int64 for dtype in idx.dtypes)

    return idx


def check_df(df: pd.DataFrame):
    # TODO(test): test it
    # logger.info("Starting check_df")

    # make sure we have a two level index
    assert df.index.nlevels == 2

    # make sure we have an int64 index
    assert all(dtype == np.int64 for dtype in df.index.dtypes.to_numpy())

    # make sure the index levels are named correctly
    assert df.index.names == ["entry", "subentry"]

    # make sure the full index is unique
    # logger.debug("make sure the full index is unique")
    # assert df.index.is_unique
    # time-consuming and not needed as a non unique index would fail the tests below

    # check the zeroth level does not repeat values in separate groups
    check_zero_level_values_for_repeated_values(
        zero_level_values=df.index.get_level_values(level=0).to_numpy()
    )

    reconstructed_event_n_jets = (
        reconstruct_event_n_jets_from_groupby_zeroth_level_values(df=df)
    )

    # make sure the 0th level index is zero based with a step size of 1
    expected_zero_level_values = get_zero_level_values_from_event_n_jets(
        event_n_jets=reconstructed_event_n_jets
    )
    assert np.array_equal(
        df.index.get_level_values(level=0),
        expected_zero_level_values,
    )
    del expected_zero_level_values

    # make sure the 1st level index is zero based within each element of the 0th level
    expected_first_level_values = get_first_level_values_from_event_n_jets_njit(
        event_n_jets=reconstructed_event_n_jets
    )
    # logger.debug("comparing")
    assert np.array_equal(
        df.index.get_level_values(level=1),
        expected_first_level_values,
    )
    del expected_first_level_values

    # logger.debug("Done with check_df")


def get_zero_level_values_from_event_n_jets(event_n_jets: np.ndarray):
    assert isinstance(event_n_jets, np.ndarray)
    assert event_n_jets.ndim == 1
    if len(event_n_jets) == 0:
        v = np.array([], dtype="int64")
    else:
        v = np.repeat(np.arange(len(event_n_jets)), event_n_jets)

    return v


@njit
def get_first_level_values_from_event_n_jets_njit(event_n_jets):
    v = np.array([i for n_jets in event_n_jets for i in range(n_jets)], dtype="int64")

    return v


def reconstruct_event_n_jets_from_groupby_zeroth_level_values(df: pd.DataFrame):
    if len(df) == 0:
        return np.array([])

    zero_level_values = df.index.get_level_values(level=0).to_numpy()

    check_zero_level_values_for_repeated_values(zero_level_values=zero_level_values)

    reconstructed_event_n_jets = df.groupby(level=0, sort=False).size().to_numpy()

    return reconstructed_event_n_jets


def check_zero_level_values_for_repeated_values(zero_level_values: np.ndarray):
    # check that values in the zero level index only appear once as a
    # group and do not reappear somewhere else outside of this group
    # e.g. want to detect the repeated 0 here
    # [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 4, 4, 4]
    #                                         ^
    # detect "islands" of a single value or a group of duplicate values in two ways

    if len(zero_level_values) != 0:
        # get last element of each group
        idx_1 = np.nonzero(np.diff(zero_level_values, append=zero_level_values[-1] - 1))
        s_1 = pd.Series(zero_level_values[idx_1])
        # get first element of each group
        idx_2 = np.nonzero(np.diff(zero_level_values, prepend=zero_level_values[0] - 1))
        s_2 = pd.Series(zero_level_values[idx_2])
        # sanity check to make sure the two ways yield the same result
        assert s_1.equals(s_2)

        value_counts = s_1.value_counts()
        if (value_counts != 1).any():
            raise ValueError(
                "A zero level value appeared in more than one group: "
                f"{value_counts[value_counts != 1]}"
            )
