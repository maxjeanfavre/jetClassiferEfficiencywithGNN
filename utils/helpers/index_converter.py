import numpy as np


class IndexConverter:
    def __init__(self):
        pass

    @staticmethod
    def get_jet_boolean_mask_from_event_boolean_mask(
        event_boolean_mask: np.ndarray, event_n_jets: np.ndarray
    ):
        if not isinstance(event_boolean_mask, np.ndarray):
            raise ValueError(
                "'event_boolean_mask' has to be np.ndarray. "
                f"Got type: {type(event_boolean_mask)}"
            )
        if event_boolean_mask.dtype != bool:
            raise ValueError(
                "event_boolean_mask has to be np.ndarray with dtype bools. "
                f"Got np.ndarray with dtype: {event_boolean_mask.dtype}"
            )
        if event_boolean_mask.ndim != 1:
            raise ValueError(
                "'event_boolean_mask' has to be 1D np.ndarray. "
                f"Got shape: {event_boolean_mask.shape}"
            )
        if not isinstance(event_n_jets, np.ndarray):
            raise ValueError(
                "'event_n_jets' has to be np.ndarray. "
                f"Got type: {type(event_n_jets)}"
            )
        if event_n_jets.dtype != int:
            raise ValueError(
                "event_n_jets has to be np.ndarray with dtype int. "
                f"Got np.ndarray with dtype: {event_n_jets.dtype}"
            )
        if event_n_jets.ndim != 1:
            raise ValueError(
                "'event_n_jets' has to be 1D np.ndarray. "
                f"Got shape: {event_n_jets.shape}"
            )
        if len(event_boolean_mask) != len(event_n_jets):
            raise ValueError(
                "'event_boolean_mask' and 'event_n_jets' must have the same length. "
                "Had lengths: "
                f"'event_boolean_mask': {len(event_boolean_mask)}, "
                f"'event_n_jets': {len(event_n_jets)}"
            )

        jet_boolean_mask = np.repeat(event_boolean_mask, event_n_jets)

        return jet_boolean_mask

    @staticmethod
    def get_event_boolean_mask_from_event_index(event_index: np.ndarray, n_events: int):
        if not isinstance(event_index, np.ndarray):
            raise ValueError(
                f"'event_index' has to be np.ndarray. Got type: {type(event_index)}"
            )
        if event_index.dtype != int:
            raise ValueError(
                "event_index has to be np.ndarray with dtype int."
                f"Got np.ndarray with dtype: {event_index.dtype}"
            )
        if event_index.ndim != 1:
            raise ValueError(
                "'event_index' has to be 1D np.ndarray. "
                f"Got shape: {event_index.shape}"
            )
        if len(np.unique(event_index)) != len(event_index):
            raise ValueError(
                "'event_index' contains duplicate values. "
                "This leads to discrepancies between using "
                "event_index and the event_boolean_mask"
            )
        if not isinstance(n_events, int):
            raise ValueError(f"'n_events' has to be int. Got type: {type(n_events)}")
        if np.any(np.diff(event_index) <= 0):
            raise ValueError(
                "event_index is not strictly monotonically increasing. "
                "This leads to different orders when using the event_index "
                "compared to the resulting event_boolean_mask"
            )
        if np.any(event_index >= n_events):
            raise ValueError(
                "event_index had at least one entry inconsistent with n_events. "
                f"Got: n_events: {n_events}, "
                "inconsistent unique values of event_index: "
                f"{set(event_index[event_index >= n_events])}"
            )

        event_boolean_mask = np.full(shape=n_events, fill_value=False, dtype=bool)
        event_boolean_mask[event_index] = True

        return event_boolean_mask
