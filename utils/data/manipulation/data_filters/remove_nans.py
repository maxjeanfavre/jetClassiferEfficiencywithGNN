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


class RemoveNaNs(DataFilter):
    def __init__(
        self,
        active_modes: Tuple[str, ...],
        filter_full_event: bool,
    ) -> None:
        if filter_full_event:
            description = "Removes events where at least one jet has any missing values"
        else:
            description = "Removes jets with any missing values"

        super().__init__(
            name=__class__.__name__,
            description=description,
            active_modes=active_modes,
        )

        self.filter_full_event = filter_full_event

    def _manipulate_inplace(self, jds: JetEventsDataset) -> None:
        logger.trace("Determining columns with nan values")
        cols_with_nan_values = {
            col: nan_values
            for col, nan_values in jds.df.isna().sum(axis=0).iteritems()
            if nan_values > 0
        }
        if cols_with_nan_values:
            logger.trace(
                "Found nan values. "
                f"Columns and number of nan values: {cols_with_nan_values}"
            )
            logger.trace("Creating jet_boolean_mask")
            jet_boolean_mask = ~jds.df.isna().any(
                axis=1
            )  # rows without nan values get True

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
                reconstruct_event_n_jets_from_groupby_zeroth_level_values(
                    df=df_manipulated
                )
            )

            logger.trace("Getting new index")
            new_idx = get_idx_from_event_n_jets(event_n_jets=event_n_jets_after_filters)

            logger.trace("Setting new index")
            df_manipulated.index = new_idx

            logger.trace("Setting df_manipulated into JetEventsDataset")
            jds.df = df_manipulated
        else:
            logger.trace("Found no nan values")

        assert jds.df.isna().sum().sum() == 0

    def required_columns(self) -> Tuple[str, ...]:
        return tuple()

    def added_columns(self) -> Tuple[str, ...]:
        return tuple()
