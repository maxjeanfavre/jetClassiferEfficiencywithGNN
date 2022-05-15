from __future__ import annotations

import pathlib
from typing import List, Tuple

import pandas as pd

from utils.configs.working_point import WorkingPointConfig
from utils.configs.working_points_set import WorkingPointsSetConfig
from utils.data.jet_events_dataset import JetEventsDataset


class Model:
    def __init__(
        self, working_points_set_config: WorkingPointsSetConfig, *args, **kwargs
    ) -> None:
        self.working_points_set_config = working_points_set_config

    def save(self, path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path) -> Model:
        # TODO(test): test whether the loaded model makes the same predictions as the one still in memory
        raise NotImplementedError

    def train(
        self, jds: JetEventsDataset, path_to_save: pathlib.Path, **kwargs
    ) -> None:
        raise NotImplementedError

    def predict(
        self,
        jds: JetEventsDataset,
        working_point_configs: List[WorkingPointConfig],
        **kwargs
    ) -> List[Tuple[pd.Series, pd.Series]]:
        raise NotImplementedError

    def get_required_columns(self) -> Tuple[str, ...]:
        raise NotImplementedError
