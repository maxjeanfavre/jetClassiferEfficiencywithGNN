from __future__ import annotations

from typing import Tuple

import numpy as np
from loguru import logger

from utils.data.dataframe_format import get_idx_from_event_n_jets
from utils.data.jet_events_dataset import JetEventsDataset
from utils.data.manipulation.data_filters.data_filter import DataFilter


class FirstNJetsFilter(DataFilter):
    def __init__(
        self,
        active_modes: Tuple[str, ...],
        n: int,
    ) -> None:
        if n == 0:
            raise ValueError(f"n can't be zero. Got: {n}")
        super().__init__(
            name=__class__.__name__,
            description=f"Keeps the first {n} jets in each event",
            active_modes=active_modes,
        )
        self.n = n

    def _manipulate_inplace(self, jds: JetEventsDataset) -> None:
        assert self.n != 0

        logger.trace("Getting df_manipulated with .loc")
        df_manipulated = jds.df.loc[(slice(None), slice(0, self.n - 1)), :]

        logger.trace("Getting event_n_jets")
        event_n_jets_after_filters = np.minimum(jds.event_n_jets, self.n)
        # event_n_jets_after_filters_old = (
        #     reconstruct_event_n_jets_from_groupby_zeroth_level_values(df=df_manipulated)
        # )
        # assert np.array_equal(
        #     a1=event_n_jets_after_filters,
        #     a2=event_n_jets_after_filters_old,
        # )

        logger.trace("Getting new index")
        new_idx = get_idx_from_event_n_jets(event_n_jets=event_n_jets_after_filters)

        logger.trace("Setting new index")
        df_manipulated.index = new_idx

        logger.trace("Setting df_manipulated into JetEventsDataset")
        jds.df = df_manipulated

    def required_columns(self) -> Tuple[str, ...]:
        required_columns = tuple()
        return required_columns

    def added_columns(self) -> Tuple[str, ...]:
        added_columns = tuple()
        return added_columns
