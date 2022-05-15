from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from loguru import logger

from utils.data.dataframe_format import (
    get_idx_from_event_n_jets,
)
from utils.data.manipulation.data_filters.data_filter import DataFilter
from utils.exceptions import UnexpectedError
from utils.helpers.index_converter import IndexConverter

if TYPE_CHECKING:
    from utils.data.jet_events_dataset import JetEventsDataset


class JetMultiplicityFilter(DataFilter):
    def __init__(
        self,
        active_modes: Tuple[str, ...],
        n: int,
        mode: str,
    ) -> None:
        if n == 0:
            raise ValueError(f"n can't be zero. Got: {n}")
        if mode == "keep":
            description = f"Removes Events that do not have a jet multiplicity of {n}"
        elif mode == "remove":
            description = f"Removes Events with a jet multiplicity of {n}"
        else:
            raise ValueError(f"Unsupported value for 'mode'. Got: {mode}")

        super().__init__(
            name=__class__.__name__,
            description=description,
            active_modes=active_modes,
        )
        self.n = n
        self.mode = mode

    def _manipulate_inplace(self, jds: JetEventsDataset) -> None:
        logger.trace("Creating event_boolean_mask")
        if self.mode == "keep":
            event_boolean_mask = jds.event_n_jets == self.n
        elif self.mode == "remove":
            event_boolean_mask = jds.event_n_jets != self.n
        else:
            raise UnexpectedError(
                f"Unsupported value for self.mode. It was: {self.mode}"
            )

        logger.trace("Converting event_boolean_mask to jet_boolean_mask")
        jet_boolean_mask = IndexConverter.get_jet_boolean_mask_from_event_boolean_mask(
            event_boolean_mask=event_boolean_mask, event_n_jets=jds.event_n_jets
        )

        logger.trace("Getting df_manipulated with .loc")
        df_manipulated = jds.df.loc[jet_boolean_mask]

        logger.trace("Getting event_n_jets")
        event_n_jets_after_filters = jds.event_n_jets[event_boolean_mask]
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
