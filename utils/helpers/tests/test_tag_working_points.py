from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from utils.configs.working_point import WorkingPointConfig
from utils.configs.working_points_set import WorkingPointsSetConfig
from utils.data.dataframe_format import get_idx_from_event_n_jets
from utils.data.jet_events_dataset import JetEventsDataset
from utils.helpers.tag_working_points import get_tag_working_points


class TestGetTagWorkingPoints:
    @pytest.mark.parametrize("n_jets_per_event", [(1, 1), (1, 2), (1, 30)])
    @pytest.mark.parametrize("n_events", [1, 10, 10 ** 2, 10 ** 4])
    def test_get_tag_working_points(
        self, n_events: int, n_jets_per_event: Tuple[int, int]
    ):
        event_n_jets = np.random.randint(
            n_jets_per_event[0], n_jets_per_event[1] + 1, n_events
        )

        n_jets = np.sum(event_n_jets)

        df = pd.DataFrame(
            data={"Jet_btagDeepFlavB": np.random.random(size=n_jets)},
            index=get_idx_from_event_n_jets(event_n_jets=event_n_jets),
        )

        jds = JetEventsDataset(df=df)

        working_points = {
            "btagWP_Loose_DeepCSV": 0.1241,
            "btagWP_Medium_DeepCSV": 0.4184,
            "btagWP_Tight_DeepCSV": 0.7527,
        }

        working_points_set_config = WorkingPointsSetConfig(
            name="standard_wps_set",
            working_points=tuple(
                WorkingPointConfig(
                    name=name,
                    expression=f"Jet_btagDeepFlavB > {value}",
                    required_columns=("Jet_btagDeepFlavB",),
                )
                for name, value in working_points.items()
            ),
        )

        tag = get_tag_working_points(
            jds=jds,
            working_points_set_config=working_points_set_config,
        )

        tag_manual = []
        for value in df["Jet_btagDeepFlavB"].to_numpy():
            tag_ = 0
            for i, cut_off_value in enumerate(working_points.values()):
                if value > cut_off_value:
                    tag_ = i + 1
            tag_manual.append(tag_)
        tag_manual = np.array(tag_manual)

        np.testing.assert_array_equal(
            x=tag,
            y=tag_manual,
        )
