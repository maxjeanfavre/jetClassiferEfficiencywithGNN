from __future__ import annotations

import pathlib
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.configs.working_point import WorkingPointConfig
from utils.configs.working_points_set import WorkingPointsSetConfig
from utils.data.jet_events_dataset import JetEventsDataset
from utils.efficiency_map.binned_efficiency_maps import BinnedEfficiencyMaps
from utils.models.model import Model


class BinnedEfficiencyMapModel(Model):
    model_filename = "model.pkl"
    bem_filename = "binned_efficiency_maps.pkl"

    def __init__(
        self,
        working_points_set_config: WorkingPointsSetConfig,
        bins: Dict[str, np.ndarray],
        separation_cols: Tuple[str, ...],
    ) -> None:
        super().__init__(working_points_set_config=working_points_set_config)

        self.bins = bins
        self.separation_cols = separation_cols

        self.bem = None

    def save(self, path) -> None:
        with open(path / self.model_filename, "wb") as f:
            pickle.dump(self, f)
        self.bem.save(path=path / self.bem_filename)

    @classmethod
    def load(cls, path) -> BinnedEfficiencyMapModel:
        with open(path / cls.model_filename, "rb") as f:
            model = pickle.load(f)
            model: BinnedEfficiencyMapModel
        model.bem = BinnedEfficiencyMaps.load(path=path / model.bem_filename)
        return model

    def train(
        self, jds: JetEventsDataset, path_to_save: pathlib.Path, **kwargs
    ) -> None:
        self.bem = BinnedEfficiencyMaps.create_efficiency_maps(
            jds=jds,
            bins=self.bins,
            separation_cols=self.separation_cols,
            working_points_set_config=self.working_points_set_config,
        )

    def predict(
        self, jds: JetEventsDataset, working_point_configs: List[WorkingPointConfig]
    ) -> List[Tuple[pd.Series, pd.Series]]:
        # make sure the model was trained with the requested working points
        for working_point_config in working_point_configs:
            assert working_point_config in self.working_points_set_config.working_points

        predictions = []

        self.bem: BinnedEfficiencyMaps

        for working_point_config in working_point_configs:
            print("compute efficiency of working_point_config.name :",working_point_config.name)
            results, err = self.bem.compute_efficiency(
                jds=jds, working_point_name=working_point_config.name
            )

            predictions.append((results, err))

        return predictions

    def get_required_columns(self) -> Tuple[str, ...]:
        required_columns = set()
        required_columns.update(self.working_points_set_config.get_required_columns())
        required_columns.update(self.bins.keys())
        required_columns.update(self.separation_cols)
        required_columns = tuple(required_columns)
        return required_columns
