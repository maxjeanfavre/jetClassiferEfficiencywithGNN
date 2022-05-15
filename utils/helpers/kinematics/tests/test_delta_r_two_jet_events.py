import numpy as np
import pytest

from utils.helpers.kinematics.delta_r_two_jet_events import (
    compute_delta_r_two_jet_events,
)
from utils.helpers.kinematics.tests.test_delta_r import compute_delta_r_old


class TestDeltaRTwoJetEvents:
    @pytest.mark.parametrize("n_events", [1, 10, 10 ** 2, 10 ** 4])
    def test_delta_r_two_jet_events(self, n_events):
        event_n_jets = np.random.randint(2, 2 + 1, n_events)
        n_rows = np.sum(event_n_jets)

        eta = np.random.random(size=n_rows)
        phi = np.random.random(size=n_rows)

        delta_r_two_jet_events = compute_delta_r_two_jet_events(
            event_n_jets=event_n_jets, eta=eta, phi=phi
        )

        delta_r_one_by_one = []
        for i in range(n_events):
            eta_1 = eta[i * 2 + 0]
            eta_2 = eta[i * 2 + 1]
            phi_1 = phi[i * 2 + 0]
            phi_2 = phi[i * 2 + 1]

            delta_r = compute_delta_r_old(
                eta_1=eta_1, phi_1=phi_1, eta_2=eta_2, phi_2=phi_2
            )
            delta_r_one_by_one.append(delta_r)

        np.testing.assert_array_equal(
            x=delta_r_two_jet_events,
            y=delta_r_one_by_one,
        )

    def test_errors(self):
        with pytest.raises(ValueError):
            compute_delta_r_two_jet_events(
                event_n_jets=[1],
                eta=np.array([1]),
                phi=np.array([1]),
            )
        with pytest.raises(ValueError):
            compute_delta_r_two_jet_events(
                event_n_jets=np.array([1]),
                eta=[1],
                phi=np.array([1]),
            )
        with pytest.raises(ValueError):
            compute_delta_r_two_jet_events(
                event_n_jets=np.array([1]),
                eta=np.array([1]),
                phi=[1],
            )

        with pytest.raises(ValueError):
            compute_delta_r_two_jet_events(
                event_n_jets=np.array([1]),
                eta=np.array([1]),
                phi=np.array([1]),
            )

        n_events = 10 ** 3
        event_n_jets = np.full(shape=n_events * 2, fill_value=2)
        n_jets = 2 * n_events
        with pytest.raises(ValueError):
            compute_delta_r_two_jet_events(
                event_n_jets=event_n_jets,
                eta=np.random.random(size=n_jets - 1),
                phi=np.random.random(size=n_jets),
            )
        with pytest.raises(ValueError):
            compute_delta_r_two_jet_events(
                event_n_jets=event_n_jets,
                eta=np.random.random(size=n_jets),
                phi=np.random.random(size=n_jets - 1),
            )
        with pytest.raises(ValueError):
            compute_delta_r_two_jet_events(
                event_n_jets=event_n_jets,
                eta=np.concatenate(
                    (np.random.random(size=n_jets - 1), np.array([np.nan]))
                ),
                phi=np.random.random(size=n_jets),
            )
        with pytest.raises(ValueError):
            compute_delta_r_two_jet_events(
                event_n_jets=event_n_jets,
                eta=np.random.random(size=n_jets),
                phi=np.concatenate(
                    (np.random.random(size=n_jets - 1), np.array([np.nan]))
                ),
            )
