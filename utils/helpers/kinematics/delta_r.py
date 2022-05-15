import numpy as np
from numba import njit


def compute_delta_r(event_n_jets: np.ndarray, eta: np.ndarray, phi: np.ndarray):
    for name, arr in [["event_n_jets", event_n_jets], ["eta", eta], ["phi", phi]]:
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"'{name}' is not an np.ndarray")
    if np.any(event_n_jets == 0):
        raise ValueError("'event_n_jets' contained entries of 0")
    n_jets = np.sum(event_n_jets)
    if len(eta) != n_jets:
        raise ValueError(
            f"'eta' length mismatch. Got length: {len(eta)}. "
            f"Expected length: {n_jets}"
        )
    if len(phi) != n_jets:
        raise ValueError(
            f"'phi' length mismatch. Got length: {len(phi)}. "
            f"Expected length: {n_jets}"
        )

    delta_r_values = compute_delta_r_njit(
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
def compute_delta_r_njit(event_n_jets: np.ndarray, eta: np.ndarray, phi: np.ndarray):
    delta_r_values = np.full(
        shape=np.sum(event_n_jets * (event_n_jets - 1)),
        fill_value=np.nan,
        dtype=np.float64,
    )

    events_jets_offset = np.concatenate((np.array([0]), np.cumsum(event_n_jets[:-1])))

    running_idx = 0
    for event_idx, n_jets in enumerate(event_n_jets):
        n_jets_offset = events_jets_offset[event_idx]
        for primary_jet_idx in range(n_jets):
            for secondary_jet_idx in range(n_jets):
                if primary_jet_idx != secondary_jet_idx:
                    delta_r = compute_delta_r_single_combination_njit(
                        eta_1=eta[n_jets_offset + primary_jet_idx],
                        phi_1=phi[n_jets_offset + primary_jet_idx],
                        eta_2=eta[n_jets_offset + secondary_jet_idx],
                        phi_2=phi[n_jets_offset + secondary_jet_idx],
                    )
                    delta_r_values[running_idx] = delta_r
                    running_idx += 1

    return delta_r_values


@njit
def compute_delta_r_single_combination_njit(
    eta_1: float, phi_1: float, eta_2: float, phi_2: float
):
    d_eta = eta_1 - eta_2
    d_phi = phi_1 - phi_2

    while d_phi >= np.pi:
        d_phi -= 2 * np.pi
    while d_phi < -np.pi:
        d_phi += 2 * np.pi

    return np.sqrt(d_eta * d_eta + d_phi * d_phi)


# edge_dict has to be in the same order as delta_r is calculated
def get_edge_dict(max_n):
    edge_dict = {}

    for n in range(1, max_n + 1):
        ind = np.arange(n ** 2) % (n + 1) != 0
        src = np.repeat(np.arange(n), n)[ind]
        dst = np.tile(np.arange(n), n)[ind]
        edge_dict[n] = [src, dst]

    return edge_dict
