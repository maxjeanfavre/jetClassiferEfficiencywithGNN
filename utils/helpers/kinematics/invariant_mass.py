import numpy as np


def compute_invariant_mass(
    pt_1: np.ndarray,
    pt_2: np.ndarray,
    eta_1: np.ndarray,
    eta_2: np.ndarray,
    phi_1: np.ndarray,
    phi_2: np.ndarray,
):
    for a, name in [
        [pt_1, "pt_1"],
        [pt_2, "pt_2"],
        [eta_1, "eta_1"],
        [eta_2, "eta_2"],
        [phi_1, "phi_1"],
        [phi_2, "phi_2"],
    ]:
        if not isinstance(a, np.ndarray):
            raise ValueError(f"{name} has to be 'np.ndarray'. Got type: {type(a)}")
        if a.ndim != 1:
            raise ValueError(f"{name} has to be 1D array. Got shape: {a.shape}")

    res = np.sqrt(2 * pt_1 * pt_2 * (np.cosh(eta_1 - eta_2) - np.cos(phi_1 - phi_2)))

    return res
