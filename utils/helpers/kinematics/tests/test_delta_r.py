from typing import Tuple

import numpy as np
import pytest

from utils.helpers.kinematics.delta_r import (
    compute_delta_r,
    compute_delta_r_single_combination_njit,
    get_edge_dict,
)


def delta_phi(x):
    # while len(np.where(x >= np.pi)[0]) > 0:
    #     x[np.where(x >= np.pi)[0]] -= 2 * np.pi
    # while len(np.where(x < -np.pi)[0]) > 0:
    #     x[np.where(x < -np.pi)[0]] += 2 * np.pi

    while np.sum(x >= np.pi) > 0:
        x[x >= np.pi] -= 2 * np.pi
    while np.sum(x < -np.pi) > 0:
        x[x < -np.pi] += 2 * np.pi

    return x


def compute_delta_r_old(eta_1, phi_1, eta_2, phi_2):
    d_eta = eta_1 - eta_2
    d_phi = delta_phi(phi_1 - phi_2)

    return np.sqrt(d_eta * d_eta + d_phi * d_phi)


class TestComputeDeltaR:
    @pytest.mark.parametrize("n_jets_per_event", [(1, 1), (1, 2), (1, 30)])
    @pytest.mark.parametrize("n_events", [1, 10, 10 ** 2, 10 ** 4, 10 ** 5])
    def test_compute_delta_r(self, n_events: int, n_jets_per_event: Tuple[int, int]):
        # this test also makes sure that the order of jet combinations in the
        # delta_r caclulcation is the same order as given by get_edge_dict
        event_n_jets = np.random.randint(
            n_jets_per_event[0], n_jets_per_event[1] + 1, n_events
        )
        n_jets = np.sum(event_n_jets)

        eta = np.random.random(size=n_jets)
        phi = np.random.random(size=n_jets)

        delta_r = compute_delta_r(event_n_jets=event_n_jets, eta=eta, phi=phi)

        assert len(delta_r) == np.sum(event_n_jets * (event_n_jets - 1))

        edge_dict = get_edge_dict(max_n=n_jets_per_event[1])

        events_jets_offset = np.cumsum(
            np.concatenate((np.array([0]), event_n_jets[:-1]))
        )  # value at index i gives number of jets in events 0, ..., i - 1 combined

        delta_r_old = []
        for i in range(len(event_n_jets)):
            n_jets_event = event_n_jets[i]
            a = np.vstack(edge_dict[n_jets_event])
            jets_offset = events_jets_offset[i]
            delta_r_old_ = compute_delta_r_old(
                eta_1=eta[a[0] + jets_offset],
                phi_1=phi[a[0] + jets_offset],
                eta_2=eta[a[1] + jets_offset],
                phi_2=phi[a[1] + jets_offset],
            )
            delta_r_old.append(delta_r_old_)

        delta_r_old = np.concatenate(delta_r_old)

        np.testing.assert_array_equal(
            x=delta_r,
            y=delta_r_old,
        )

    def test_errors(self):
        with pytest.raises(ValueError):
            compute_delta_r(
                event_n_jets=[1],
                eta=np.array([1]),
                phi=np.array([1]),
            )
        with pytest.raises(ValueError):
            compute_delta_r(
                event_n_jets=np.array([1]),
                eta=[1],
                phi=np.array([1]),
            )
        with pytest.raises(ValueError):
            compute_delta_r(
                event_n_jets=np.array([1]),
                eta=np.array([1]),
                phi=[1],
            )

        with pytest.raises(ValueError):
            compute_delta_r(
                event_n_jets=np.array([0]),
                eta=np.array([1]),
                phi=np.array([1]),
            )

        n_events = 10 ** 3
        n_jets_per_event = (1, 10)
        event_n_jets = np.random.randint(
            n_jets_per_event[0], n_jets_per_event[1] + 1, n_events
        )
        n_jets = np.sum(event_n_jets)
        with pytest.raises(ValueError):
            compute_delta_r(
                event_n_jets=event_n_jets,
                eta=np.random.random(size=n_jets - 1),
                phi=np.random.random(size=n_jets),
            )
        with pytest.raises(ValueError):
            compute_delta_r(
                event_n_jets=event_n_jets,
                eta=np.random.random(size=n_jets),
                phi=np.random.random(size=n_jets - 1),
            )
        with pytest.raises(ValueError):
            compute_delta_r(
                event_n_jets=event_n_jets,
                eta=np.concatenate(
                    (np.random.random(size=n_jets - 1), np.array([np.nan]))
                ),
                phi=np.random.random(size=n_jets),
            )
        with pytest.raises(ValueError):
            compute_delta_r(
                event_n_jets=event_n_jets,
                eta=np.random.random(size=n_jets),
                phi=np.concatenate(
                    (np.random.random(size=n_jets - 1), np.array([np.nan]))
                ),
            )


class TestComputeDeltaRSingleCombinationNjit:
    @pytest.mark.parametrize("n", [1, 10, 10 ** 2, 10 ** 4])
    def test___compute_delta_r_single_combination_njit(self, n: int):
        eta_1 = np.random.random(size=n)
        phi_1 = np.random.random(size=n)
        eta_2 = np.random.random(size=n)
        phi_2 = np.random.random(size=n)

        for eta_1_, phi_1_, eta_2_, phi_2_ in zip(eta_1, phi_1, eta_2, phi_2):
            delta_r_njit = compute_delta_r_single_combination_njit(
                eta_1=eta_1_,
                phi_1=phi_1_,
                eta_2=eta_2_,
                phi_2=phi_2_,
            )

            delta_r_old = compute_delta_r_old(
                eta_1=eta_1_,
                phi_1=phi_1_,
                eta_2=eta_2_,
                phi_2=phi_2_,
            )

            assert delta_r_njit == delta_r_old
