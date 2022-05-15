from __future__ import annotations

import time
from typing import TYPE_CHECKING, Tuple

from loguru import logger

if TYPE_CHECKING:
    from utils.data.jet_events_dataset import JetEventsDataset


class DataManipulator:
    def __init__(
        self,
        name: str,
        description: str,
        active_modes: Tuple[str, ...],
    ) -> None:
        self.name = name
        self.description = description
        self.active_modes = active_modes

    def manipulate_inplace(self, jds: JetEventsDataset) -> None:
        logger.debug(f"Data Manipulator: {self.name} starting. {self.description}")

        s = time.time()
        self._manipulate_inplace(jds=jds)
        e = time.time()

        logger.debug(f"Data Manipulator: {self.name} done in {e - s:.2f} seconds")

    def _manipulate_inplace(self, jds: JetEventsDataset) -> None:
        raise NotImplementedError

    def required_columns(self) -> Tuple[str, ...]:
        raise NotImplementedError

    def added_columns(self) -> Tuple[str, ...]:
        raise NotImplementedError
