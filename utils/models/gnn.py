from __future__ import annotations

import pathlib
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple, Type

import dgl
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.configs.working_point import WorkingPointConfig
from utils.configs.working_points_set import WorkingPointsSetConfig
from utils.data.jet_events_dataset import JetEventsDataset
from utils.helpers.hash_dataframe import hash_df
from utils.helpers.kinematics.delta_r import get_edge_dict
from utils.helpers.kinematics.delta_r_split_up_by_event import (
    get_delta_r_split_up_by_event,
)
from utils.helpers.tag_working_points import get_tag_working_points
from utils.helpers.torch_tensor_to_numpy import convert_torch_tensor_to_numpy
from utils.models.gnn_jet_efficiency_net import JetEfficiencyNet
from utils.models.model import Model
from utils.preprocessing.pipeline import PreprocessingPipeline

NODE_FEATURES = "node_features"
FLAVOUR_INDICES = "flav_indices"
DELTA_R = "dR"


class GNN(Model):
    model_init_args_filename = "model_init_args.pkl"
    torch_filename = "model.pt"
    preprocessing_pipeline_filename = "preprocessing_pipeline.pkl"

    def __init__(
        self,
        working_points_set_config: WorkingPointsSetConfig,
        node_features_cols: List[str],
        flavour_col: str,
        preprocessing_pipeline: PreprocessingPipeline,
        feats,
        correction_layers,
        flavour_embedding_num_embeddings,
        flavour_embedding_dim,
        flavour_index_conversion_dict,
        edge_network_dropout,
        node_network_dropout,
        eff_correction_dropout,
        old_mode: bool = False,
        old_mode_wp_idx: Optional[int] = None,
    ) -> None:
        super().__init__(working_points_set_config=working_points_set_config)

        self.node_features_cols = node_features_cols
        self.flavour_col = flavour_col
        self.preprocessing_pipeline = preprocessing_pipeline
        self.feats = feats
        self.correction_layers = correction_layers
        self.flavour_embedding_num_embeddings = flavour_embedding_num_embeddings
        self.flavour_embedding_dim = flavour_embedding_dim
        self.flavour_index_conversion_dict = flavour_index_conversion_dict
        self.edge_network_dropout = edge_network_dropout
        self.node_network_dropout = node_network_dropout
        self.eff_correction_dropout = eff_correction_dropout
        self.old_mode = old_mode

        if self.old_mode:
            n_classes = 1
            if old_mode_wp_idx is None:
                raise ValueError(
                    "If 'old_mode' is used, 'old_mode_wp_idx' can't be None"
                )
            if not isinstance(old_mode_wp_idx, int):
                raise ValueError(
                    "'old_mode_wp_idx' has to be 'int'. "
                    f"Got type: {type(old_mode_wp_idx)}"
                )

            if old_mode_wp_idx < 1 or old_mode_wp_idx > len(
                self.working_points_set_config
            ):
                raise ValueError(
                    "'old_mode_wp_idx' has to be an integer in "
                    f"[1, {len(self.working_points_set_config)}]"
                )
        else:
            n_classes = len(self.working_points_set_config) + 1
            if old_mode_wp_idx is not None:
                raise ValueError(
                    "'old_mode_wp_idx' can't be not None if 'old_mode' is not used"
                )

        self.old_mode_wp_idx = old_mode_wp_idx

        self.estimator = JetEfficiencyNet(
            in_features=len(self.node_features_cols) + self.flavour_embedding_dim,
            feats=self.feats,
            correction_layers=self.correction_layers,
            num_classes=n_classes,
            flavour_embedding_num_embeddings=self.flavour_embedding_num_embeddings,
            flavour_embedding_dim=self.flavour_embedding_dim,
            edge_network_dropout=self.edge_network_dropout,
            node_network_dropout=self.node_network_dropout,
            eff_correction_dropout=self.eff_correction_dropout,
        )

    def save(self, path) -> None:
        if isinstance(self.estimator, nn.DataParallel):
            torch.save(self.estimator.module.state_dict(), path / self.torch_filename)
        else:
            torch.save(self.estimator.state_dict(), path / self.torch_filename)

        self.preprocessing_pipeline.save(
            path=path / self.preprocessing_pipeline_filename
        )

        model_init_args = {
            "working_points_set_config": self.working_points_set_config,
            "node_features_cols": self.node_features_cols,
            "flavour_col": self.flavour_col,
            "feats": self.feats,
            "correction_layers": self.correction_layers,
            "flavour_embedding_num_embeddings": self.flavour_embedding_num_embeddings,
            "flavour_embedding_dim": self.flavour_embedding_dim,
            "flavour_index_conversion_dict": self.flavour_index_conversion_dict,
            "edge_network_dropout": self.edge_network_dropout,
            "node_network_dropout": self.node_network_dropout,
            "eff_correction_dropout": self.eff_correction_dropout,
            "old_mode": self.old_mode,
            "old_mode_wp_idx": self.old_mode_wp_idx,
        }

        with open(path / self.model_init_args_filename, "wb") as f:
            pickle.dump(model_init_args, f)

    @classmethod
    def load(cls, path) -> GNN:
        with open(path / cls.model_init_args_filename, "rb") as f:
            model_init_args = pickle.load(f)

        preprocessing_pipeline = PreprocessingPipeline.load(
            path=path / cls.preprocessing_pipeline_filename
        )
        model = cls(preprocessing_pipeline=preprocessing_pipeline, **model_init_args)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if isinstance(
        #     model.estimator, nn.DataParallel
        # ):
        #     model.estimator.module.load_state_dict(
        #         torch.load(path / model.torch_filename, map_location=device)
        #     )
        # else:
        #     model.estimator.load_state_dict(
        #         torch.load(path / model.torch_filename, map_location=device)
        #     )
        model.estimator.load_state_dict(
            torch.load(path / model.torch_filename, map_location=device)
        )

        model.estimator.to(device)

        return model

    def train(
        self,
        jds: JetEventsDataset,
        path_to_save: pathlib.Path,
        epochs: int = 1,
        batch_size: int = 128,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_cls_init_kwargs: Dict[str, Any] = None,
        learning_rate_scheduler_cls: Type[
            torch.optim.lr_scheduler
        ] = torch.optim.lr_scheduler.LambdaLR,
        learning_rate_scheduler_cls_init_kwargs: Dict[str, Any] = None,
        train_split=0.8,
        validation_split=0.2,
        loss_func: Optional[torch.nn.modules.loss._Loss] = None,
    ) -> None:
        if optimizer_cls_init_kwargs is None:
            optimizer_cls_init_kwargs = {}
        if learning_rate_scheduler_cls_init_kwargs is None:
            learning_rate_scheduler_cls_init_kwargs = {"lr_lambda": lambda epoch: 1}
        if loss_func is not None:
            logger.warning(
                "A loss function was supplied. Please make sure "
                "overwriting the default choice is intended."
                f"The supplied loss function was: {loss_func}"
            )

        if train_split + validation_split != 1:
            raise ValueError(
                "'train_split' and 'validation_split' have to add up to 1. "
                f"Got: 'train_split': {train_split}, "
                f"'validation_split': {validation_split}"
            )

        jds_train, jds_validation = jds.split_data(
            train_size=train_split,
            test_size=validation_split,
            return_train=True,
            return_test=True,
            random_state=42,
            copy=True,
        )

        del jds

        # calculate delta_r values here, so they are not calculated
        # with preprocessed eta and phi values
        delta_r_split_up_by_event_train = get_delta_r_split_up_by_event(
            event_n_jets=jds_train.event_n_jets,
            eta=jds_train.df["Jet_eta"].to_numpy(),
            phi=jds_train.df["Jet_phi"].to_numpy(),
        )
        delta_r_split_up_by_event_validation = get_delta_r_split_up_by_event(
            event_n_jets=jds_validation.event_n_jets,
            eta=jds_validation.df["Jet_eta"].to_numpy(),
            phi=jds_validation.df["Jet_phi"].to_numpy(),
        )

        # calculate flavour_index here with the unprocessed data
        flavour_index_train = get_flavour_index(
            flavour_data=jds_train.df[self.flavour_col].to_numpy(),
            conversion_dict=self.flavour_index_conversion_dict,
        )
        flavour_index_validation = get_flavour_index(
            flavour_data=jds_validation.df[self.flavour_col].to_numpy(),
            conversion_dict=self.flavour_index_conversion_dict,
        )

        # calculate target here with the unprocessed data
        target_train = get_tag_working_points(
            jds=jds_train,
            working_points_set_config=self.working_points_set_config,
        )
        target_validation = get_tag_working_points(
            jds=jds_validation,
            working_points_set_config=self.working_points_set_config,
        )

        if self.old_mode:
            # modify target
            target_train = (target_train >= self.old_mode_wp_idx).astype("int64")
            target_validation = (target_validation >= self.old_mode_wp_idx).astype(
                "int64"
            )

        self.preprocessing_pipeline.fit(df=jds_train.df)

        jds_train_preprocessed = JetEventsDataset(
            df=self.preprocessing_pipeline.transform(
                df=jds_train.df,
            )
        )

        del jds_train

        jds_validation_preprocessed = JetEventsDataset(
            df=self.preprocessing_pipeline.transform(
                df=jds_validation.df,
            )
        )

        del jds_validation

        node_feature_col_names_preprocessed = (
            self.preprocessing_pipeline.get_new_col_name(
                col_name=self.node_features_cols
            )
        )
        flavour_col_name_preprocessed = self.preprocessing_pipeline.get_new_col_name(
            col_name=self.flavour_col
        )
        assert len(flavour_col_name_preprocessed) == 1
        flavour_col_name_preprocessed = flavour_col_name_preprocessed[0]

        logger.debug(
            "jds_train_preprocessed.df.memory_usage(deep=True).sum() / (1024 ** 3): "
            f"{jds_train_preprocessed.df.memory_usage(deep=True).sum() / (1024 ** 3)}"
        )
        logger.debug(
            "jds_validation_preprocessed.df.memory_usage(deep=True).sum() / (1024 ** 3): "
            f"{jds_validation_preprocessed.df.memory_usage(deep=True).sum() / (1024 ** 3)}"
        )

        logger.trace(
            "hash jds_train_preprocessed.df: "
            f"{hash_df(df=jds_train_preprocessed.df)}"
        )
        logger.trace(
            "hash jds_validation_preprocessed.df: "
            f"{hash_df(df=jds_validation_preprocessed.df)}"
        )

        self.estimator.train()  # set in torch train mode

        logger.debug(f"torch.version.cuda: {torch.version.cuda}")
        logger.debug(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        logger.debug(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

        use_gpu = torch.cuda.is_available()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.debug(f"Using torch.device: {device}")
        if use_gpu:
            # if torch.cuda.device_count() > 1:
            #     if not isinstance(self.estimator, nn.DataParallel):
            #         self.estimator = nn.DataParallel(self.estimator)
            logger.debug(f"Number of GPUs: {torch.cuda.device_count()}")
        self.estimator.to(device)

        dataloader_train = self.get_dataloader(
            jds=jds_train_preprocessed,
            delta_r_split_up_by_event=delta_r_split_up_by_event_train,
            flavour_index=flavour_index_train,
            node_features_cols=node_feature_col_names_preprocessed,
            flavour_col=flavour_col_name_preprocessed,
            target=target_train,
            pin_memory=use_gpu,
            shuffle=True,
            batch_size=batch_size,
            preload_dataset=True,
        )

        del jds_train_preprocessed

        dataloader_validation = self.get_dataloader(
            jds=jds_validation_preprocessed,
            delta_r_split_up_by_event=delta_r_split_up_by_event_validation,
            flavour_index=flavour_index_validation,
            node_features_cols=node_feature_col_names_preprocessed,
            flavour_col=flavour_col_name_preprocessed,
            target=target_validation,
            pin_memory=use_gpu,
            shuffle=False,
            batch_size=batch_size,
            preload_dataset=True,
        )

        del jds_validation_preprocessed

        optimizer = optimizer_cls(
            self.estimator.parameters(), **optimizer_cls_init_kwargs
        )
        learning_rate_scheduler = learning_rate_scheduler_cls(
            optimizer, **learning_rate_scheduler_cls_init_kwargs
        )
        if loss_func is None:
            if self.old_mode:
                loss_func = nn.BCEWithLogitsLoss(reduction="mean")
            else:
                loss_func = nn.CrossEntropyLoss(reduction="mean")

        writer = SummaryWriter(
            log_dir=str((path_to_save / "tensorboard_logs").resolve())
        )

        global_step = 0

        for epoch in range(epochs):
            writer.add_scalar(
                tag="epoch",
                scalar_value=epoch,
                global_step=global_step,
            )

            epoch_train_loss = 0

            learning_rate = optimizer.param_groups[0]["lr"]
            writer.add_scalar(
                tag="optimizer learning rate",
                scalar_value=optimizer.param_groups[0]["lr"],
                global_step=epoch * len(dataloader_train),
            )
            assert learning_rate == learning_rate_scheduler.get_last_lr()[0]

            self.estimator.train()

            for x, _, target in tqdm(dataloader_train, file=sys.stdout):
                if use_gpu:
                    x = x.to(device)
                    target = target.to(device)

                output = self.estimator(x)

                if self.old_mode:
                    loss = loss_func(output, target.reshape(-1, 1).float())
                else:
                    loss = loss_func(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()

                grad_sum = 0
                for param in self.estimator.parameters():
                    grad_sum += param.grad.pow(2).sum()
                writer.add_scalar(
                    tag="grad_sum",
                    scalar_value=grad_sum,
                    global_step=global_step,
                )
                del grad_sum

                global_step += 1

            learning_rate_scheduler.step()

            epoch_train_loss /= len(dataloader_train)

            writer.add_scalar(
                tag="epoch train loss",
                scalar_value=epoch_train_loss,
                global_step=global_step,
            )

            epoch_validation_loss = 0

            self.estimator.eval()

            with torch.no_grad():
                for x, _, target in tqdm(dataloader_validation, file=sys.stdout):
                    if use_gpu:
                        x = x.to(device)
                        target = target.to(device)

                    output = self.estimator(x)

                    if self.old_mode:
                        loss = loss_func(output, target.reshape(-1, 1).float())
                    else:
                        loss = loss_func(output, target)

                    epoch_validation_loss += loss.item()

            epoch_validation_loss /= len(dataloader_validation)

            writer.add_scalar(
                tag="epoch validation loss",
                scalar_value=epoch_validation_loss,
                global_step=global_step,
            )

            logger.info(
                f"epoch: {epoch}, "
                f"epoch train loss: {epoch_train_loss}, "
                f"epoch validation loss: {epoch_validation_loss}, "
                f"learning rate: {learning_rate}"
            )

            # TODO(medium): save model with the lowest epoch validation loss

    def predict(
        self,
        jds: JetEventsDataset,
        working_point_configs: List[WorkingPointConfig],
        batch_size: int = 32,
        return_raw_outputs: bool = False,
    ) -> List[Tuple[pd.Series, pd.Series]]:
        # make sure the model was trained with the requested working points
        for working_point_config in working_point_configs:
            assert working_point_config in self.working_points_set_config.working_points

            if self.old_mode:
                # if the model is trained with only one wp,
                # make sure that one was requested
                working_point_set_idx = (
                    self.working_points_set_config.working_points.index(
                        working_point_config
                    )
                )

                if working_point_set_idx + 1 != self.old_mode_wp_idx:
                    raise ValueError(
                        "Model was not trained for the requested working point: "
                        f"{working_point_config.name}"
                    )

        # calculate delta_r values here, so they are not calculated
        # with preprocessed eta and phi values
        delta_r_split_up_by_event = get_delta_r_split_up_by_event(
            event_n_jets=jds.event_n_jets,
            eta=jds.df["Jet_eta"].to_numpy(),
            phi=jds.df["Jet_phi"].to_numpy(),
        )

        # calculate flavour_index here with the unprocessed data
        flavour_index = get_flavour_index(
            flavour_data=jds.df[self.flavour_col].to_numpy(),
            conversion_dict=self.flavour_index_conversion_dict,
        )

        target = np.full(shape=jds.n_jets, fill_value=0, dtype="int64")

        jds_preprocessed = JetEventsDataset(
            df=self.preprocessing_pipeline.transform(
                df=jds.df,
                only_cols=[
                    *self.node_features_cols,
                    self.flavour_col,
                ],
            ),
        )

        del jds

        node_feature_col_names_preprocessed = (
            self.preprocessing_pipeline.get_new_col_name(
                col_name=self.node_features_cols
            )
        )
        flavour_col_name_preprocessed = self.preprocessing_pipeline.get_new_col_name(
            col_name=self.flavour_col
        )
        assert len(flavour_col_name_preprocessed) == 1
        flavour_col_name_preprocessed = flavour_col_name_preprocessed[0]

        use_gpu = torch.cuda.is_available()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.debug(f"Using torch.device: {device}")
        if use_gpu:
            logger.debug(f"Number of GPUs: {torch.cuda.device_count()}")

        self.estimator.to(device)

        dataloader = self.get_dataloader(
            jds=jds_preprocessed,
            delta_r_split_up_by_event=delta_r_split_up_by_event,
            flavour_index=flavour_index,
            node_features_cols=node_feature_col_names_preprocessed,
            flavour_col=flavour_col_name_preprocessed,
            target=target,
            pin_memory=use_gpu,
            shuffle=False,
            batch_size=batch_size,
            preload_dataset=False,
        )

        df_results = pd.DataFrame()
        df_results.index = jds_preprocessed.df.index

        self.estimator.eval()

        raw_outputs = []

        with torch.no_grad():
            for _, (x, _, _) in enumerate(tqdm(dataloader, file=sys.stdout)):
                if use_gpu:
                    x = x.to(device)

                raw_output_batch = self.estimator(x)

                raw_outputs.append(raw_output_batch)

        raw_outputs = torch.cat(raw_outputs, dim=0)

        assert raw_outputs.shape[0] == jds_preprocessed.n_jets

        if return_raw_outputs:
            return raw_outputs
        else:
            predictions = []

            for working_point_config in working_point_configs:
                if self.old_mode:
                    output_activation_func = torch.nn.Sigmoid()
                    output_activated = output_activation_func(raw_outputs)

                    assert output_activated.dim() == 2
                    assert output_activated.shape[0] == jds_preprocessed.n_jets
                    assert output_activated.shape[1] == 1
                    output = output_activated.flatten()
                else:
                    output_activation_func = torch.nn.Softmax(dim=1)
                    output_activated = output_activation_func(raw_outputs)

                    working_point_set_idx = (
                        self.working_points_set_config.working_points.index(
                            working_point_config
                        )
                    )
                    wp_sum_idx = working_point_set_idx + 1
                    output = output_activated[:, wp_sum_idx:].sum(dim=1)
                    assert output.dim() == 1
                    assert output.shape[0] == jds_preprocessed.n_jets

                output = convert_torch_tensor_to_numpy(output)

                results = pd.Series(output, dtype="float64")
                results.index = jds_preprocessed.df.index

                err = pd.Series(
                    data=np.nan, index=jds_preprocessed.df.index, dtype="float64"
                )

                predictions.append((results, err))

            return predictions

    def get_required_columns(self) -> Tuple[str, ...]:
        required_columns = []
        required_columns.extend(["Jet_eta", "Jet_phi"])  # for delta_r calculations
        required_columns.extend(self.node_features_cols)
        required_columns.append(self.flavour_col)

        required_columns = tuple(required_columns)

        return required_columns

    @staticmethod
    def get_dataloader(
        jds: JetEventsDataset,
        delta_r_split_up_by_event: List[np.ndarray],
        flavour_index: np.ndarray,
        node_features_cols: List[str],
        flavour_col: str,
        target: np.ndarray,
        pin_memory: bool,
        shuffle: bool,
        batch_size: int,
        preload_dataset: bool,
    ):
        dataset = TorchDataset(
            jds=jds,
            node_features_cols=node_features_cols,
            flavour_col=flavour_col,
            target=target,
            flavour_index=flavour_index,
            delta_r=delta_r_split_up_by_event,
            pin_memory=pin_memory,
            preload=preload_dataset,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collate,
            pin_memory=False,
        )

        return dataloader


def get_flavour_index(flavour_data: np.ndarray, conversion_dict: Dict):
    if not isinstance(flavour_data, np.ndarray):
        raise ValueError(
            "flavour_data has to be instance of np.ndarray. "
            f"Got type: {type(flavour_data)}"
        )
    if flavour_data.ndim != 1:
        raise ValueError(
            f"flavour_data has to be 1D np.ndarray. Had dim {flavour_data.ndim}"
        )

    res = np.full(shape=len(flavour_data), fill_value=np.nan)

    for flavour, flavour_index in conversion_dict.items():
        res[flavour_data == flavour] = flavour_index

    assert not np.any(np.isnan(res))

    res = res.astype("int")

    return res


class TorchDataset(Dataset):
    def __init__(
        self,
        jds: JetEventsDataset,
        node_features_cols: List[str],
        flavour_col: str,
        target: np.ndarray,
        flavour_index: np.ndarray,
        delta_r: List[np.ndarray],
        pin_memory: bool,
        preload: bool,
    ) -> None:
        assert type(flavour_col) == str

        assert type(target) == np.ndarray
        assert pd.api.types.is_integer_dtype(target)
        assert target.ndim == 1
        assert len(target) == jds.n_jets

        assert type(flavour_index) == np.ndarray
        assert pd.api.types.is_integer_dtype(flavour_index)
        assert flavour_index.ndim == 1
        assert len(flavour_index) == jds.n_jets

        assert type(delta_r) == list
        assert all(isinstance(i, np.ndarray) for i in delta_r)
        assert all(i.ndim == 1 for i in delta_r)
        assert np.array_equal(
            np.array([len(i) for i in delta_r]),
            (jds.event_n_jets * (jds.event_n_jets - 1)),
        )

        self.preload = preload

        self.node_features_tensor = torch.FloatTensor(
            jds.df[node_features_cols].to_numpy()
        )
        self.flavour_tensor = torch.FloatTensor(jds.df[flavour_col].to_numpy())
        self.flavour_index_tensor = torch.LongTensor(flavour_index)
        self.target_tensor = torch.FloatTensor(target)

        assert self.flavour_tensor.dim() == 1
        assert self.flavour_index_tensor.dim() == 1
        assert self.target_tensor.dim() == 1

        self.delta_r = [torch.FloatTensor(arr) for arr in delta_r]

        if pin_memory:
            self.node_features_tensor = self.node_features_tensor.pin_memory()
            self.flavour_tensor = self.flavour_tensor.pin_memory()
            self.flavour_index_tensor = self.flavour_index_tensor.pin_memory()
            self.target_tensor = self.target_tensor.pin_memory()

        self.events_jets_offset = np.cumsum(
            np.concatenate((np.array([0]), jds.event_n_jets[:-1]))
        )  # value at index i gives number of jets in events 0, ..., i - 1 combined

        self.event_n_jets = jds.event_n_jets
        self.n_events = jds.n_events

        # src and dst dict
        self.edge_dict = get_edge_dict(max_n=np.max(self.event_n_jets))

        if self.preload:
            self.g_ = []
            self.flavour_ = []
            self.target_ = []
            for i in tqdm(range(self.__len__()), file=sys.stdout):
                g, flavour, target = self.__getitem__real(idx=i)
                self.g_.append(g)
                self.flavour_.append(flavour)
                self.target_.append(target)

    def __len__(self):
        # Returns number of events
        return self.n_events

    def __getitem__(self, idx):
        if self.preload:
            g = self.g_[idx]
            flavour = self.flavour_[idx]
            target = self.target_[idx]
        else:
            g, flavour, target = self.__getitem__real(idx=idx)

        return g, flavour, target

    def __getitem__real(self, idx):
        # idx is the index of an event
        jets_offset = self.events_jets_offset[idx]

        # number of jets in the requested event
        n_jets = self.event_n_jets[idx]

        # list of indices of jets in the requested event
        r = list(range(jets_offset, jets_offset + n_jets))

        node_features = self.node_features_tensor[r]
        flavour_indices = self.flavour_index_tensor[r]
        target = self.target_tensor[r]
        flavour = self.flavour_tensor[r]

        src, dst = self.edge_dict[n_jets]
        delta_r = self.delta_r[idx]
        assert delta_r.shape[0] == n_jets * (n_jets - 1)

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     g = dgl.DGLGraph()
        #
        # g.add_nodes(n_jets)
        #
        # g.ndata[NODE_FEATURES] = node_features
        # g.ndata[FLAVOUR_INDICES] = flavour_indices
        #
        # g.add_edges(u=src, v=dst)
        #
        # g.edata[DELTA_R] = delta_r

        g = dgl.graph((src, dst), num_nodes=n_jets)
        g.ndata[NODE_FEATURES] = node_features
        g.ndata[FLAVOUR_INDICES] = flavour_indices
        g.edata[DELTA_R] = delta_r

        return g, flavour, target


def collate(samples):
    graphs = [x[0] for x in samples]
    flavours = [x[1] for x in samples]
    targets = [x[2] for x in samples]

    batched_graph = dgl.batch(
        graphs=graphs, ndata=[NODE_FEATURES, FLAVOUR_INDICES], edata=[DELTA_R]
    )
    flavours = torch.cat(flavours)
    targets = torch.cat(targets)

    flavours = flavours.unsqueeze(1).float()
    targets = targets.long()

    return batched_graph, flavours, targets
