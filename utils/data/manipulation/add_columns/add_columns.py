from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from utils.data.manipulation.data_manipulator import DataManipulator
from utils.helpers.columns_not_present_yet import check_columns_not_present_yet

if TYPE_CHECKING:
    from utils.data.jet_events_dataset import JetEventsDataset


class AddColumns(DataManipulator):
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

    def _manipulate_inplace(self, jds: JetEventsDataset) -> None:
        check_columns_not_present_yet(df=jds.df, cols=self.added_columns())
        raise NotImplementedError

    def required_columns(self) -> Tuple[str, ...]:
        raise NotImplementedError

    def added_columns(self) -> Tuple[str, ...]:
        raise NotImplementedError
