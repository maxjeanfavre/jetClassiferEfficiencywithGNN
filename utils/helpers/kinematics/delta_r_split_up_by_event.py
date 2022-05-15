import numpy as np

from utils.helpers.kinematics.delta_r import compute_delta_r


def get_delta_r_split_up_by_event(
    event_n_jets: np.ndarray, eta: np.ndarray, phi: np.ndarray
):
    delta_r = compute_delta_r(
        event_n_jets=event_n_jets,
        eta=eta,
        phi=phi,
    )

    delta_r_split_up_by_event = np.split(
        delta_r,
        np.cumsum(event_n_jets * (event_n_jets - 1))[:-1],
    )

    return delta_r_split_up_by_event
