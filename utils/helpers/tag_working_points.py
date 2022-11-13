import numpy as np

from utils.configs.working_points_set import WorkingPointsSetConfig
from utils.data.jet_events_dataset import JetEventsDataset


def get_tag_working_points(
    jds: JetEventsDataset,
    working_points_set_config: WorkingPointsSetConfig,
) -> np.ndarray:
    tag = np.full(shape=jds.n_jets, fill_value=0, dtype="int64")

    assert tag.dtype == "int64"
    assert (tag == 0).sum() == jds.n_jets

    for i, working_point_config in enumerate(working_points_set_config.working_points):
        tag[jds.df.eval(working_point_config.expression).to_numpy()] = i + 1
        #print("expression is ",working_point_config.expression, "and tag is ",tag[0:50])

    assert tag.dtype == "int64"
    assert np.isfinite(tag).all()

    return tag
