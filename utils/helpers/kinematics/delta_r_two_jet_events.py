import numpy as np
from numba import njit

from utils.helpers.kinematics.delta_r import compute_delta_r_single_combination_njit


def compute_delta_r_two_jet_events(
    event_n_jets: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
):
    for name, arr in [["event_n_jets", event_n_jets], ["eta", eta], ["phi", phi]]:
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"'{name}' is not an np.ndarray")
    if np.any(event_n_jets != 2):
        raise ValueError(
            "Can only compute delta_r value of an event "
            "if all events contain exactly two jets. "
            f"Got event_n_jets containing: {np.unique(event_n_jets)}"
        )
    if len(eta) != 2 * len(event_n_jets):
        raise ValueError(
            f"'eta' length mismatch. Got length: {len(eta)}. "
            f"Expected length: {2 * len(event_n_jets)}"
        )
    if len(phi) != 2 * len(event_n_jets):
        raise ValueError(
            f"'phi' length mismatch. Got length: {len(phi)}. "
            f"Expected length: {2 * len(event_n_jets)}"
        )

    delta_r_values = __compute_delta_r_two_jet_events_njit(
        event_n_jets=event_n_jets,
        eta=eta,
        phi=phi,
    )

    if np.any(np.isnan(delta_r_values)):
        raise ValueError(
            f"Result had {np.count_nonzero(np.isnan(delta_r_values))} 'np.nan' values"
        )

    return delta_r_values


@njit
def __compute_delta_r_two_jet_events_njit(
    event_n_jets: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
):
    # 'parallel' mode did not bring performance improvements in local tests
    assert np.all(event_n_jets == 2), "Can only handle events with exactly two jets"

    delta_r_values = np.full(
        shape=len(event_n_jets),
        fill_value=np.nan,
        dtype=np.float64,
    )

    events_jets_offset = np.concatenate((np.array([0]), np.cumsum(event_n_jets[:-1])))

    for event_idx in range(len(event_n_jets)):
        total_n_jets = events_jets_offset[event_idx]
        delta_r = compute_delta_r_single_combination_njit(
            eta_1=eta[0 + total_n_jets],
            phi_1=phi[0 + total_n_jets],
            eta_2=eta[1 + total_n_jets],
            phi_2=phi[1 + total_n_jets],
        )
        delta_r_values[event_idx] = delta_r

    return delta_r_values
