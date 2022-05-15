import numpy as np

from utils.configs.model import ModelConfig
from utils.data.manipulation.data_filters.eval_expression import EvalExpressionFilter
from utils.models.binned_efficiency_map_model import BinnedEfficiencyMapModel

model_config = ModelConfig(
    name="eff_map_pt_eta",
    data_manipulators=(
        EvalExpressionFilter(
            description=(
                "Keeps jets with valid btagDeepB value (0 <= Jet_btagDeepB <= 1)"
            ),
            active_modes=("train",),
            expression="0 <= Jet_btagDeepB <= 1",
            filter_full_event=False,
            required_columns=("Jet_btagDeepB",),
        ),
    ),
    model_cls=BinnedEfficiencyMapModel,
    model_init_kwargs={
        "bins": {
            "Jet_Pt": np.array(
                [
                    -np.inf,
                    0,
                    10,
                    30,
                    50,
                    70,
                    100,
                    150,
                    200,
                    250,
                    300,
                    350,
                    400,
                    600,
                    1000,
                    np.inf,
                ]
            ),
            "Jet_eta": np.array(
                [
                    -np.inf,
                    -2.8,
                    -2.5,
                    -2,
                    -1.5,
                    -1,
                    -0.5,
                    0,
                    0.5,
                    1,
                    1.5,
                    2,
                    2.5,
                    2.8,
                    np.inf,
                ]
            ),
        },
        "separation_cols": ("Jet_hadronFlavour",),
    },
    model_train_kwargs={},
    model_predict_kwargs={},
)
