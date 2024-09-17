import torch

from utils.configs.model import ModelConfig
from utils.data.manipulation.data_filters.eval_expression import EvalExpressionFilter
from utils.models.mass_dijet_metric.euclidian_norm import GNN
from utils.preprocessing.identity import Identity
from utils.preprocessing.normalizer import Normalizer
from utils.preprocessing.pipeline import PreprocessingPipeline


def lr_func(epoch):
    return 1


model_config = ModelConfig(
    name="gnn_edge_mass_dijet_euclidian_norm",
    data_manipulators=(
        EvalExpressionFilter(
            description=(
                "Keeps events with valid btagDeepB value (0 <= Jet_btagDeepFlavB <= 1)"
            ),
            active_modes=("train",),
            expression="0 <= Jet_btagDeepFlavB <= 1",
            filter_full_event=True,
            required_columns=("Jet_btagDeepFlavB",),
        ),
    ),
    model_cls=GNN,
    model_init_kwargs={
        "node_features_cols": [
            "Jet_Pt",
            "Jet_eta",
            "Jet_phi",
        ],
        "flavour_col": "Jet_hadronFlavour",
        "preprocessing_pipeline": PreprocessingPipeline(
            column_preprocessors={
                "Jet_Pt": Normalizer(),
                "Jet_eta": Normalizer(),
                "Jet_phi": Normalizer(),
                "Jet_hadronFlavour": Identity(),
            }
        ),
        "edge_hidden_state_sizes": [256, 256, 256, 256, 256],
        "node_hidden_state_sizes": [512, 512, 512, 512, 512],
        "jet_efficiency_net_hidden_layers": [512, 256, 128, 50],
        "flavour_embedding_num_embeddings": 3,  # only have 3 flavours (0, 4, 5)
        "flavour_embedding_dim": 2,
        "flavour_index_conversion_dict": {
            5: 0,
            4: 1,
            0: 2,
        },
        "edge_network_dropout": 0.3,
        "node_network_dropout": 0.3,
        "jet_efficiency_net_dropout": 0.3,
        "old_mode": False,
        "old_mode_wp_idx": None,
    },
    model_train_kwargs={
        "epochs": 30,
        "batch_size": 1024,
        "optimizer_cls": torch.optim.Adam,
        "optimizer_cls_init_kwargs": {"lr": 1e-4},
        "learning_rate_scheduler_cls": torch.optim.lr_scheduler.LambdaLR,
        "learning_rate_scheduler_cls_init_kwargs": {
            "lr_lambda": lr_func,
            "verbose": False,
        },
        "train_split": 0.95,
        "validation_split": 0.05,
        "loss_func": None,
    },
    model_predict_kwargs={
        "batch_size": 2048,
        "return_raw_outputs": False,
    },
)
