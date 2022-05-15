from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from utils.data.manipulation.add_columns.add_columns import AddColumns
from utils.helpers.columns_not_present_yet import check_columns_not_present_yet

if TYPE_CHECKING:
    from utils.data.jet_events_dataset import JetEventsDataset


class AddPairwiseValueSwapWithinEvent(AddColumns):
    def __init__(
        self,
        active_modes: Tuple[str, ...],
        column: str,
        new_column: str,
    ) -> None:
        self.column = column
        self.new_column = new_column
        super().__init__(
            name=__class__.__name__,
            description=(
                f"Will add the value of {self.column} of one jet in the event as "
                f"{self.new_column} in the other jet and vice versa"
            ),
            active_modes=active_modes,
        )

    def _manipulate_inplace(self, jds: JetEventsDataset) -> None:
        check_columns_not_present_yet(df=jds.df, cols=self.added_columns())

        unique_event_n_jets = np.unique(jds.event_n_jets)
        if len(unique_event_n_jets) != 1 or (unique_event_n_jets != 2).any():
            raise ValueError(
                "Can only do pairwise value swap to JetEventsDataset "
                "if all events contain two jets. "
                "Got JetEventsDataset with event_n_jets containing: "
                f"{unique_event_n_jets}."
            )

        assert np.array_equal(
            jds.df.index.get_level_values(level=1).unique().to_numpy(), np.array([0, 1])
        )
        jds.df[self.new_column] = (
            jds.df[self.column]
            .sort_index(level=[0, 1], ascending=[True, False])
            .to_numpy()
        )
        assert jds.df[self.new_column].dtype == jds.df[self.column].dtype

    def required_columns(self) -> Tuple[str, ...]:
        required_columns = (self.column,)
        return required_columns

    def added_columns(self) -> Tuple[str, ...]:
        added_columns = (self.new_column,)
        return added_columns
