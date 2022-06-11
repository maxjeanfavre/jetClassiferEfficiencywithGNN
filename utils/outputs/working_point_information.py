import json
import pathlib
from typing import Optional

from utils.configs.dataset import DatasetConfig
from utils.configs.working_points_set import WorkingPointsSetConfig
from utils.data.jet_events_dataset import JetEventsDataset


def save_working_point_information(
    dataset_config: DatasetConfig,
    working_points_set_config: WorkingPointsSetConfig,
    output_dir_path: pathlib.Path,
    jds: Optional[JetEventsDataset] = None,
):
    # TODO(low): logging
    if jds is None:
        jds = JetEventsDataset.read_in(
            dataset_config=dataset_config,
            branches=("Jet_btagDeepB", "Jet_hadronFlavour"),
        )

    data = {}

    for working_point_config in working_points_set_config.working_points:
        jet_boolean_mask = jds.df.eval(working_point_config.expression)

        n_light_jets = (jds.df["Jet_hadronFlavour"] == 0).sum()

        n_light_jets_passing_working_point = (
            jds.df.loc[jet_boolean_mask, "Jet_hadronFlavour"] == 0
        ).sum()

        light_jet_mistag_rate = n_light_jets_passing_working_point / n_light_jets

        n_b_jets = (jds.df["Jet_hadronFlavour"] == 5).sum()

        n_b_jets_passing_working_point = (
            jds.df.loc[jet_boolean_mask, "Jet_hadronFlavour"] == 5
        ).sum()

        b_jet_efficiency = n_b_jets_passing_working_point / n_b_jets

        data[working_point_config.name] = {
            "light_jet_mistag_rate": light_jet_mistag_rate,
            "b_jet_efficiency": b_jet_efficiency,
        }

    with open(
        output_dir_path / f"{working_points_set_config.name}.json",
        "w",
    ) as f:
        json.dump(obj=data, fp=f, indent=4)
