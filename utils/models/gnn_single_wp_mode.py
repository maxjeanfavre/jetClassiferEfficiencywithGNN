from __future__ import annotations

import pathlib
import pickle
from typing import List, Tuple

import pandas as pd

from utils.configs.working_point import WorkingPointConfig
from utils.configs.working_points_set import WorkingPointsSetConfig
from utils.data.jet_events_dataset import JetEventsDataset
from utils.models.gnn import GNN
from utils.models.model import Model


class GNNSingleWPMode(Model):
    model_init_args_filename = "model_init_args.pkl"

    def __init__(
        self, working_points_set_config: WorkingPointsSetConfig, **kwargs
    ) -> None:
        super().__init__(working_points_set_config=working_points_set_config)

        self.gnn_models = dict()
        if kwargs:
            if "old_mode" in kwargs:
                del kwargs["old_mode"]
            if "old_mode_wp_idx" in kwargs:
                del kwargs["old_mode_wp_idx"]
            for i, working_point_config in enumerate(
                self.working_points_set_config.working_points
            ):
                self.gnn_models[working_point_config.name] = GNN(
                    working_points_set_config=self.working_points_set_config,
                    old_mode=True,
                    old_mode_wp_idx=i + 1,
                    **kwargs,
                )

    def save(self, path) -> None:
        for working_point_name, working_point_model in self.gnn_models.items():
            working_point_path = path / working_point_name
            working_point_path.mkdir(parents=True, exist_ok=True)
            working_point_model.save(path=working_point_path)

        model_init_args = {
            "working_points_set_config": self.working_points_set_config,
        }

        with open(path / self.model_init_args_filename, "wb") as f:
            pickle.dump(model_init_args, f)

    @classmethod
    def load(cls, path) -> GNNSingleWPMode:
        with open(path / cls.model_init_args_filename, "rb") as f:
            model_init_args = pickle.load(f)

        model = GNNSingleWPMode(**model_init_args)

        for working_point_config in model.working_points_set_config.working_points:
            working_point_name = working_point_config.name
            working_point_path = path / working_point_name
            working_point_model = GNN.load(path=working_point_path)
            model.gnn_models[working_point_name] = working_point_model

        return model

    def train(
        self, jds: JetEventsDataset, path_to_save: pathlib.Path, **kwargs
    ) -> None:
        for working_point_name, working_point_model in self.gnn_models.items():
            working_point_path = path_to_save / working_point_name
            working_point_path.mkdir(parents=True, exist_ok=True)
            working_point_model.train(
                jds=jds, path_to_save=working_point_path, **kwargs
            )

    def predict(
        self,
        jds: JetEventsDataset,
        working_point_configs: List[WorkingPointConfig],
        **kwargs
    ) -> List[Tuple[pd.Series, pd.Series]]:
        predictions = []
        for working_point_config in working_point_configs:
            working_point_model = self.gnn_models[working_point_config.name]
            working_point_predictions = working_point_model.predict(
                jds=jds, working_point_configs=[working_point_config], **kwargs
            )
            assert len(working_point_predictions) == 1
            working_point_predictions = working_point_predictions[0]
            predictions.append(working_point_predictions)

        return predictions

    def get_required_columns(self) -> Tuple[str, ...]:
        required_columns = set()
        for working_point_model in self.gnn_models.values():
            required_columns.update(working_point_model.get_required_columns())

        required_columns = tuple(required_columns)

        return required_columns
