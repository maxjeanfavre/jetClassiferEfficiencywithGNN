from __future__ import annotations

from typing import Tuple

from loguru import logger

from utils.data.dataframe_format import (
    get_idx_from_event_n_jets,
    reconstruct_event_n_jets_from_groupby_zeroth_level_values,
)
from utils.data.jet_events_dataset import JetEventsDataset
from utils.data.manipulation.data_filters.data_filter import DataFilter
from utils.helpers.jet_boolean_mask_full_event import (
    convert_jet_boolean_mask_full_event,
)


class EvalExpressionFilter(DataFilter):
    # the rows where self.df.eval(expression) yields True are kept
    # if filter_full_event is set to True, any event with at least one mask
    # value of False will be removed entirely
    def __init__(
        self,
        description: str,
        active_modes: Tuple[str, ...],
        expression: str,
        filter_full_event: bool,
        required_columns: Tuple[str, ...],
    ) -> None:
        super().__init__(
            name=__class__.__name__,
            description=description,
            active_modes=active_modes,
        )
        self.expression = expression
        self.filter_full_event = filter_full_event
        self._required_columns = required_columns

    def _manipulate_inplace(self, jds: JetEventsDataset) -> None:
        logger.trace("Creating jet_boolean_mask")
        jet_boolean_mask = jds.df.eval(self.expression)

        if self.filter_full_event:
            jet_boolean_mask = convert_jet_boolean_mask_full_event(
                jet_boolean_mask=jet_boolean_mask,
                jds=jds,
            )

        logger.trace("Getting df_manipulated with .loc")
        df_manipulated = jds.df.loc[jet_boolean_mask]

        # as I started with a jds.df here I can safely use
        # reconstruct_event_jets_from_groupby_zeroth_level_values
        logger.trace("Getting event_n_jets from groupby_zeroth_level_values")
        event_n_jets_after_filters = (
            reconstruct_event_n_jets_from_groupby_zeroth_level_values(df=df_manipulated)
        )

        logger.trace("Getting new index")
        new_idx = get_idx_from_event_n_jets(event_n_jets=event_n_jets_after_filters)

        logger.trace("Setting new index")
        df_manipulated.index = new_idx

        logger.trace("Setting df_manipulated into JetEventsDataset")
        jds.df = df_manipulated

    def required_columns(self) -> Tuple[str, ...]:
        required_columns = tuple(i for i in self._required_columns)
        return required_columns

    def added_columns(self) -> Tuple[str, ...]:
        added_columns = tuple()
        return added_columns
