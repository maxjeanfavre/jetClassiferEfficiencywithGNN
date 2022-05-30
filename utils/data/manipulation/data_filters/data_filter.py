import time
from typing import Tuple

from loguru import logger

from utils.data.jet_events_dataset import JetEventsDataset
from utils.data.manipulation.data_manipulator import DataManipulator


class DataFilter(DataManipulator):
    def __init__(
        self,
        name: str,
        description: str,
        active_modes: Tuple[str, ...],
    ) -> None:
        super().__init__(
            name=f"{__class__.__name__}_{name}",
            description=description,
            active_modes=active_modes,
        )

    def manipulate_inplace(self, jds: JetEventsDataset) -> None:
        logger.debug(f"Data Manipulator: {self.name} starting. {self.description}")

        n_events_before = jds.n_events
        n_jets_before = jds.n_jets

        s = time.time()
        self._manipulate_inplace(jds=jds)
        e = time.time()

        n_events_after = jds.n_events
        n_jets_after = jds.n_jets

        logger.debug(
            f"Data Manipulator: {self.name} done in {e - s:.2f} seconds. "
            "Remaining: "
            f"{n_events_after} / {n_events_before} "
            f"({n_events_after / n_events_before:.2%}) of events, "
            f"{n_jets_after} / {n_jets_before} "
            f"({n_jets_after / n_jets_before:.2%}) of jets"
        )

    def _manipulate_inplace(self, jds: JetEventsDataset) -> None:
        raise NotImplementedError

    def required_columns(self) -> Tuple[str, ...]:
        raise NotImplementedError

    def added_columns(self) -> Tuple[str, ...]:
        raise NotImplementedError
