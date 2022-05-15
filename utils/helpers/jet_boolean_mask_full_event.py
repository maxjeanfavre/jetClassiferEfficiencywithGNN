from loguru import logger

from utils.data.jet_events_dataset import JetEventsDataset
from utils.helpers.index_converter import IndexConverter


def convert_jet_boolean_mask_full_event(jet_boolean_mask, jds: JetEventsDataset):
    # want indices of events where mask only has values True
    # equivalent to events where the inverse of mask has only False values
    logger.trace(
        "Creating event_boolean_mask from jet_boolean_mask with filter_full_event"
    )
    event_boolean_mask = (
        (~jet_boolean_mask).groupby(level=0, sort=False).sum() == 0
    ).to_numpy()
    # this is a boolean mask of events to keep because all jets
    # in these events have a True value in the jet_boolean_mask

    logger.trace("Converting event_boolean_mask to jet_boolean_mask")
    jet_boolean_mask = IndexConverter.get_jet_boolean_mask_from_event_boolean_mask(
        event_boolean_mask=event_boolean_mask, event_n_jets=jds.event_n_jets
    )

    return jet_boolean_mask
