from __future__ import annotations

import pathlib
import pickle
from typing import List, Tuple

import pandas as pd

from utils.configs.working_point import WorkingPointConfig
from utils.configs.working_points_set import WorkingPointsSetConfig
from utils.data.jet_events_dataset import JetEventsDataset
from utils.models.model import Model


class DirectTagging(Model):
    model_filename = "model.pkl"

    def __init__(self, working_points_set_config: WorkingPointsSetConfig) -> None:
        super().__init__(working_points_set_config=working_points_set_config)

    def save(self, path) -> None:
        with open(path / self.model_filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> DirectTagging:
        with open(path / cls.model_filename, "rb") as f:
            model = pickle.load(f)

        return model

    def train(
        self, jds: JetEventsDataset, path_to_save: pathlib.Path, **kwargs
    ) -> None:
        pass

    def predict(
        self,
        jds: JetEventsDataset,
        working_point_configs: List[WorkingPointConfig],
        **kwargs
    ) -> List[Tuple[pd.Series, pd.Series]]:
        # make sure the model was trained with the requested working points
        for working_point_config in working_point_configs:
            assert working_point_config in self.working_points_set_config.working_points

        predictions = []

        for working_point_config in working_point_configs:
            results = jds.df.eval(working_point_config.expression)
            results = results.astype(int)
            results.name = None

            err = pd.Series(data=0, index=jds.df.index, dtype="float64")
            print("err")
            print(err)
            print("results\n",results)
            print("jds.df\n",jds.df)

            predictions.append((results, err))

        return predictions

    def get_required_columns(self) -> Tuple[str, ...]:
        required_columns = self.working_points_set_config.get_required_columns()

        return required_columns
