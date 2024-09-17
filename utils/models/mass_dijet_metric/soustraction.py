# GNN with dijet mass edge feature (subtraction metric) 

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

from utils.paths_handler import Paths
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
from utils.models.model import Model
from utils.preprocessing.pipeline import PreprocessingPipeline

NODE_FEATURES_KEY = "NODE_FEATURES"
NODE_HIDDEN_STATE_KEY = "NODE_HIDDEN_STATE"
EDGE_FEATURE_KEY = "m_jj"
EDGE_HIDDEN_STATE_KEY = "EDGE_HIDDEN_STATE_KEY"
FLAVOUR_INDICES_KEY = "FLAVOUR_INDICES"


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
        edge_hidden_state_sizes: List[int],
        node_hidden_state_sizes: List[int],
        jet_efficiency_net_hidden_layers: List[int],
        flavour_embedding_num_embeddings: int,
        flavour_embedding_dim: int,
        flavour_index_conversion_dict,
        edge_network_dropout: float,
        node_network_dropout: float,
        jet_efficiency_net_dropout: float,
        old_mode: bool = False,
        old_mode_wp_idx: Optional[int] = None,
    ) -> None:
        super().__init__(working_points_set_config=working_points_set_config)

        self.node_features_cols = node_features_cols
        self.flavour_col = flavour_col
        self.preprocessing_pipeline = preprocessing_pipeline
        self.edge_hidden_state_sizes = edge_hidden_state_sizes
        self.node_hidden_state_sizes = node_hidden_state_sizes
        self.jet_efficiency_net_hidden_layers = jet_efficiency_net_hidden_layers
        self.flavour_embedding_num_embeddings = flavour_embedding_num_embeddings
        self.flavour_embedding_dim = flavour_embedding_dim
        self.flavour_index_conversion_dict = flavour_index_conversion_dict
        self.edge_network_dropout = edge_network_dropout
        self.node_network_dropout = node_network_dropout
        self.jet_efficiency_net_dropout = jet_efficiency_net_dropout
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
            print("n_classes ")
            if old_mode_wp_idx is not None:
                raise ValueError(
                    "'old_mode_wp_idx' can't be not None if 'old_mode' is not used"
                )

        self.old_mode_wp_idx = old_mode_wp_idx

        self.estimator = GNNTorchModel(
            n_node_features=len(self.node_features_cols),
            edge_hidden_state_sizes=self.edge_hidden_state_sizes,
            node_hidden_state_sizes=self.node_hidden_state_sizes,
            jet_efficiency_net_hidden_layers=self.jet_efficiency_net_hidden_layers,
            n_classes=n_classes,
            flavour_embedding_num_embeddings=self.flavour_embedding_num_embeddings,
            flavour_embedding_dim=self.flavour_embedding_dim,
            edge_network_dropout=self.edge_network_dropout,
            node_network_dropout=self.node_network_dropout,
            jet_efficiency_net_dropout=self.jet_efficiency_net_dropout,
            node_features_key=NODE_FEATURES_KEY,
            node_hidden_state_key=NODE_HIDDEN_STATE_KEY,
            edge_feature_key=EDGE_FEATURE_KEY,
            edge_hidden_state_key=EDGE_HIDDEN_STATE_KEY,
            flavour_indices_key=FLAVOUR_INDICES_KEY,
        )

    def save(self, path, epoch: Optional[int]= None, ischeckpoint:bool=False) -> None:
        if ischeckpoint:
            checkpoint_file = "checkpoint_epoch" + str(epoch)
            path = path / checkpoint_file
            Paths.safe_return(path, path_type="directory", mkdir=True) 

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
            "edge_hidden_state_sizes": self.edge_hidden_state_sizes,
            "node_hidden_state_sizes": self.node_hidden_state_sizes,
            "jet_efficiency_net_hidden_layers": self.jet_efficiency_net_hidden_layers,
            "flavour_embedding_num_embeddings": self.flavour_embedding_num_embeddings,
            "flavour_embedding_dim": self.flavour_embedding_dim,
            "flavour_index_conversion_dict": self.flavour_index_conversion_dict,
            "edge_network_dropout": self.edge_network_dropout,
            "node_network_dropout": self.node_network_dropout,
            "jet_efficiency_net_dropout": self.jet_efficiency_net_dropout,
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
        trained_model_path:Optional[str] = None,
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

        mass_dijet_train = get_mass_dijet(
            event_n_jets=jds_train.event_n_jets,
            mass=jds_train.df["Jet_mass"].to_numpy(),
        )
        #Jet_mass_train = np.array(jds_train.df["Jet_mass"].to_numpy())
        #Jet_mass_train = np.array(Jet_mass_train)
        #Jet_mass_train = [np.array(a) for a in Jet_mass_train]

        mass_dijet_validation = get_mass_dijet(
            event_n_jets=jds_validation.event_n_jets,
            mass=jds_validation.df["Jet_mass"].to_numpy(),
        )
        #Jet_mass_validation = np.array(jds_validation.df["Jet_mass"].to_numpy())
        #Jet_mass_validation = [np.array(a) for a in Jet_mass_validation]
        #Jet_mass_validation = np.array(Jet_mass_validation)

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
        #print("target_train : ",target_train[0:60])
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
        
        #for i in delta_r_split_up_by_event_train:
        #    logger.debug(
        #            "i = " f"{i} " ", i.dim = " f"{i.ndim}"  #" i.len = " f"{i.len}"# " and " f"{type(Jet_mass_train[i])}"
        #            " and " f"{Jet_mass_train[i]}"
        #        )
                
        logger.debug(
            "mass_dijet_train " f"{len(mass_dijet_train)}" " and " f"{len(mass_dijet_train[-1])}"
            " and " f"{len(mass_dijet_train[-2])}" " and " f"{len(mass_dijet_train[-3])}"
            " and " f"{len(mass_dijet_train[4])}" " and " f"{len(mass_dijet_train[5])}"
            " and " f"{len(mass_dijet_train[6])}" " and " f"{len(mass_dijet_train[7])}"
            " and " f"{len(mass_dijet_train[8])}" " and " f"{len(mass_dijet_train[9])}"
            #" and " f"{len(event_n_jets)}"
        )

        logger.debug(
            "delta_r_split_train type " f"{len(delta_r_split_up_by_event_train)}" " and "
            f"{len(delta_r_split_up_by_event_train[-1])}" " and "
            f"{len(delta_r_split_up_by_event_train[-2])}" " and "
            f"{len(delta_r_split_up_by_event_train[-3])}" " and "
            f"{len(delta_r_split_up_by_event_train[4])}" " and "
            f"{len(delta_r_split_up_by_event_train[5])}" " and "
            f"{len(delta_r_split_up_by_event_train[6])}" " and "
            f"{len(delta_r_split_up_by_event_train[7])}" " and "
            f"{len(delta_r_split_up_by_event_train[8])}" " and "
            f"{len(delta_r_split_up_by_event_train[9])}"
        )

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
        if trained_model_path:
            self.estimator.load_state_dict(
                torch.load(trained_model_path, map_location=device)
            )

        #model.estimator.to(device)
        self.estimator.to(device)

        dataloader_train = self.get_dataloader(
            jds=jds_train_preprocessed,
            delta_r_split_up_by_event=delta_r_split_up_by_event_train,
            #Jet_mass=Jet_mass_train,
            mass_dijet = mass_dijet_train,
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
            #Jet_mass=Jet_mass_validation,
            mass_dijet = mass_dijet_validation,
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
                #print("x going in tqdm ",x)  Graph(num_nodes=4696, num_edges=20138,                                                                                                                                                      ndata_schemes={'NODE_FEATURES': Scheme(shape=(3,), dtype=torch.float32), 'FLAVOUR_INDICES': Scheme(shape=(), dtype=torch.int64)}                                                                       edata_schemes={'dR': Scheme(shape=(), dtype=torch.float32)})
                if use_gpu:
                    x = x.to(device)
                    target = target.to(device)

                output = self.estimator(x)

                if self.old_mode:
                    loss = loss_func(output, target.reshape(-1, 1).float())
                else:
                    loss = loss_func(output, target)
                    print("output: ",output[:5])
                    print("target: ",target[:5])

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
            if (epoch%5==0):
                self.save(path_to_save, epoch=epoch, ischeckpoint=True)
                print("saving the checkpoint no: ",epoch) 
            # TODO(low): save model with the lowest epoch validation loss,
            #  won't make a big difference as the validation loss did not
            #  increase towards the end of the training

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

        mass_dijet = get_mass_dijet(
            event_n_jets=jds.event_n_jets,
            mass=jds.df["Jet_mass"].to_numpy(),
        )
        #Jet_mass = np.array(jds.df["Jet_mass"].to_numpy())
        #Jet_mass = [np.array(a) for a in Jet_mass]

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
            mass_dijet = mass_dijet,
            #Jet_mass=Jet_mass,
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
        #print("first set of raw outpyts ",raw_outputs)        

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
                    #print("raw_outputs ",raw_outputs[:50])
                    output_activated = output_activation_func(raw_outputs)
                    #print("output_activated ",output_activated[:50])

                    working_point_set_idx = (
                        self.working_points_set_config.working_points.index(
                            working_point_config
                        )
                    )
                    #print("working_point_config ",working_point_config)
                    #print("working_point_set_idx ",working_point_set_idx)
                    wp_sum_idx = working_point_set_idx + 1
                    output = output_activated[:, wp_sum_idx:].sum(dim=1)
                    #print("output_activated[:, wp_sum_idx:] ",output_activated[:50, wp_sum_idx:])
                    #print("output_activated[:, wp_sum_idx:].sum(dim=1) ",output_activated[:50, wp_sum_idx:].sum(dim=1))
                    #print("output ",output[:50])
                    #print("wp_sum_idx was ",wp_sum_idx)
                    #print("****************************************************")
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
        required_columns.extend(["Jet_mass"]) #for mass_dijet calculations
        required_columns.extend(self.node_features_cols)
        required_columns.append(self.flavour_col)

        required_columns = tuple(required_columns)

        return required_columns

    @staticmethod
    def get_dataloader(
        jds: JetEventsDataset,
        delta_r_split_up_by_event: List[np.ndarray],
        mass_dijet: List[np.array],
        #Jet_mass: List[np.array],
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
            mass_dijet=mass_dijet,
            #Jet_mass=Jet_mass,
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


# calculate mass_dijet
def get_mass_dijet(event_n_jets: np.ndarray, mass: np.ndarray):
    for name, arr in [["event_n_jets", event_n_jets], ["mass", mass]]:
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"'{name}' is not an np.ndarray")
    if np.any(event_n_jets == 0):
        raise ValueError("'event_n_jets' contained entries of 0")
    n_jets = np.sum(event_n_jets)
    if len(mass) != n_jets:
            raise ValueError(
            f"'mass' length mismatch. Got length: {len(mass)}. "
            f"Expected length: {n_jets}"
        )

    events_jets_offset = np.concatenate((np.array([0]), np.cumsum(event_n_jets[:-1])))
    running_idx = 0
    
    mass_dijet = np.full(
        shape=np.sum(event_n_jets * (event_n_jets - 1)),
        fill_value=np.nan,
        dtype=np.float64,
    )
    for event_idx, n_jets in enumerate(event_n_jets):
        n_jets_offset = events_jets_offset[event_idx]
        for primary_jet_idx in range(n_jets):
            for secondary_jet_idx in range(n_jets):
                if primary_jet_idx != secondary_jet_idx:
                    #function to link node-node mass (can be modify)
                    mass_dijet_values = mass[n_jets_offset + primary_jet_idx] - mass[n_jets_offset + secondary_jet_idx]
                    
                    mass_dijet[running_idx] = mass_dijet_values
                    running_idx += 1
    
    if np.any(np.isnan(mass_dijet)):
        raise ValueError(
            f"Result had {np.count_nonzero(np.isnan(mass_dijet))} 'np.nan' values"
        )

    mass_dijet = np.split(
        mass_dijet,
        np.cumsum(event_n_jets * (event_n_jets - 1))[:-1],
    )
    
    return mass_dijet


class TorchDataset(Dataset):
    def __init__(
        self,
        jds: JetEventsDataset,
        node_features_cols: List[str],
        flavour_col: str,
        target: np.ndarray,
        flavour_index: np.ndarray,
        delta_r: List[np.ndarray],
        mass_dijet: List[np.array],
        #Jet_mass: List[np.array],
        #Jet_mass: str,
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

        assert type(mass_dijet) == list
        assert all(isinstance(i, np.ndarray) for i in mass_dijet)
        assert all(i.ndim == 1 for i in mass_dijet)

        assert np.array_equal(
            np.array([len(i) for i in mass_dijet]),
            np.array([len(i) for i in delta_r]),
        )

        assert np.array_equal(  #this assertion seems to be wrong but I don't understand why
            np.array([len(i) for i in mass_dijet]),
            (jds.event_n_jets * (jds.event_n_jets - 1)),
        )
        
        #assert type(Jet_mass) == np.ndarray
        #assert pd.api.types.is_integer_dtype(Jet_mass)
        #assert Jet_mass.ndim == 1
        #assert len(Jet_mass) == jds.n_jets
        
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
        self.mass_dijet = [torch.FloatTensor(arr) for arr in mass_dijet]
        #self.Jet_mass = [torch.FloatTensor(arr) for arr in Jet_mass]
        #self.Jet_mass_tensor = torch.FloatTensor(jds.df[Jet_mass].to_numpy())
        #self.Jet_mass_tensor = torch.LongTensor(Jet_mass)
        #assert self.Jet_mass_tensor.dim() == 1

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
        #self.edge_dict = get_edge_dict(max_n=np.max(self.event_n_jets))
        self.edge_dict = get_edge_dict(max_n=np.max(self.event_n_jets))

        if self.preload:
            self.g_ = []
            self.flavour_ = []
            self.target_ = []
            for i in tqdm(range(self.__len__()), file=sys.stdout):
                g, flavour, target = self.__getitem__single_event(idx=i)
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
            g, flavour, target = self.__getitem__single_event(idx=idx)

        return g, flavour, target

    def __getitem__single_event(self, idx):
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
        mass_dijet = self.mass_dijet[idx]
        assert mass_dijet.shape[0] == n_jets * (n_jets - 1)

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     g = dgl.DGLGraph()
        #
        # g.add_nodes(n_jets)
        #
        # g.ndata[NODE_FEATURES_KEY] = node_features
        # g.ndata[FLAVOUR_INDICES_KEY] = flavour_indices
        #
        # g.add_edges(u=src, v=dst)
        #
        # g.edata[EDGE_FEATURE_KEY] = delta_r

        g = dgl.graph((src, dst), num_nodes=n_jets)
        g.ndata[NODE_FEATURES_KEY] = node_features
        g.ndata[FLAVOUR_INDICES_KEY] = flavour_indices
        g.edata[EDGE_FEATURE_KEY] = mass_dijet
        #g.edata[EDGE_FEATURE_KEY] = delta_r
        #print("src ",src)
        #print("dst ",dst)
        #print("g ",g)
        #print("target ",target)
        #print("flavour ",flavour)
        #print("g.ndata[NODE_FEATURES_KEY] ",g.ndata[NODE_FEATURES_KEY])
        #print("g.ndata[NODE_FEATURES_KEY].shape ",g.ndata[NODE_FEATURES_KEY].shape)
        #print("g.edata[EDGE_FEATURE_KEY] ",g.edata[EDGE_FEATURE_KEY])

        #src  [0 0 0 1 1 1 2 2 2 3 3 3]                                                                                                                                                                         dst  [1 2 3 0 2 3 0 1 3 0 1 2]                                                                                                                                                                         g  Graph(num_nodes=4, num_edges=12,                                                                                                                                                                          ndata_schemes={'NODE_FEATURES': Scheme(shape=(3,), dtype=torch.float32), 'FLAVOUR_INDICES': Scheme(shape=(), dtype=torch.int64)}                                                                       edata_schemes={'dR': Scheme(shape=(), dtype=torch.float32)})                                                                                                                                     target  tensor([3., 0., 0., 2.])                                                                                                                                                                       flavour  tensor([5., 0., 0., 5.])   
        #src  [0 1]                                                                                                                                                                                             dst  [1 0]                                                                                                                                                                                             g  Graph(num_nodes=2, num_edges=2,                                                                                                                                                                           ndata_schemes={'NODE_FEATURES': Scheme(shape=(3,), dtype=torch.float32), 'FLAVOUR_INDICES': Scheme(shape=(), dtype=torch.int64)}                                                                       edata_schemes={'dR': Scheme(shape=(), dtype=torch.float32)})                                                                                                                                     g.ndata[NODE_FEATURES_KEY]  tensor([[ 1.1988, -0.9034, -1.4863],                                                                                                                                               [ 0.7420, -0.6154,  0.1990]])                                                                                                                                                                  g.ndata[NODE_FEATURES_KEY].shape  torch.Size([2, 3])                                                                                                                                                   g.edata[EDGE_FEATURE_KEY]  tensor([3.0938, 3.0938])  
        return g, flavour, target


def collate(samples):
    #print("samples.shape is ",samples)
    graphs = [x[0] for x in samples]
    flavours = [x[1] for x in samples]
    targets = [x[2] for x in samples]
    #print("graph in collate ",graphs) #Graph(num_nodes=3, num_edges=6,                                                                                                          ndata_schemes={'NODE_FEATURES': Scheme(shape=(3,), dtype=torch.float32), 'FLAVOUR_INDICES': Scheme(shape=(), dtype=torch.int64)}                                                                       edata_schemes={'dR': Scheme(shape=(), dtype=torch.float32)}), Graph(num_nodes=2, num_edges=2,
    #print("flavors in collate ",flavours) #flavors in collate  [tensor([0., 5.]), tensor([0., 5., 5., 0., 0.]), tensor([5., 0., 0., 0., 5., 0., 0., 0., 0.]),..

    batched_graph = dgl.batch(
        graphs=graphs,
        ndata=[NODE_FEATURES_KEY, FLAVOUR_INDICES_KEY],
        edata=[EDGE_FEATURE_KEY],
    )
    flavours = torch.cat(flavours)
    targets = torch.cat(targets)

    flavours = flavours.unsqueeze(1).float()
    targets = targets.long()
    #print("batched graph",batched_graph)
    #print("flavours afetr collate ",flavours)
    #batched graph Graph(num_nodes=4696, num_edges=20138,                                                                                                                                                         ndata_schemes={'NODE_FEATURES': Scheme(shape=(3,), dtype=torch.float32), 'FLAVOUR_INDICES': Scheme(shape=(), dtype=torch.int64)}                                                                       edata_schemes={'dR': Scheme(shape=(), dtype=torch.float32)})                                                                                                                                     flavours afetr collate  tensor([[0.],                                                                                                                                                                          [5.],                                                                                                                                                                                                  [0.],                                                                                                                                                                                                  ...,                                                                                                                                                                                                   [0.],                                                                                                                                                                                                  [0.],                                                                                                                                                                                                  [0.]])  
    return batched_graph, flavours, targets


class EdgeNetwork(nn.Module):
    def __init__(
        self,
        input_node_features_size: int,
        input_node_hidden_state_size: int,
        output_edge_hidden_state_size: int,
        dropout: float,
        first_layer: bool,
        node_features_key: str,
        node_hidden_state_key: str,
        edge_feature_key: str,
        edge_hidden_state_key: str,
    ) -> None:
        super(EdgeNetwork, self).__init__()

        if first_layer is True and input_node_hidden_state_size != 0:
            raise ValueError(
                "For the first layer the node hidden state does not exist yet "
                "and therefore input_node_hidden_state_size should be 0. "
                f"Got {input_node_hidden_state_size = }"
            )

        self.input_node_features_size = input_node_features_size
        self.input_node_hidden_state_size = input_node_hidden_state_size
        self.output_edge_hidden_state_size = output_edge_hidden_state_size
        self.dropout = dropout
        self.first_layer = first_layer
        self.node_features_key = node_features_key
        self.node_hidden_state_key = node_hidden_state_key
        self.edge_feature_key = edge_feature_key
        self.edge_hidden_state_key = edge_hidden_state_key

        # the in_features come from
        # - for both the source and destination node of an edge
        #   - 'input_node_features_size'
        #   - 'input_node_hidden_state_size' if first_layer is False
        # - a single edge feature
        in_features = (
            2 * (self.input_node_features_size + self.input_node_hidden_state_size) + 1
        )

        out_features = self.output_edge_hidden_state_size

        mid_features = int(
            (
                2 * (self.input_node_features_size + self.input_node_hidden_state_size)
                + out_features
            )
            / 2
        )
        #print("out_features  for edge network is ",out_features)
        self.net = nn.Sequential(
            nn.Linear(
                in_features=in_features,
                out_features=mid_features,
                bias=True,
            ),
            nn.Dropout(p=self.dropout),
            nn.ReLU(),
            nn.Linear(
                in_features=mid_features,
                out_features=out_features,
                bias=True,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        #print("x is ",x)
        #print(type(x))
        #print("forward pass of edge update called ")
        if self.first_layer is True:
            input_data_list = [
                x.dst[self.node_features_key],
                x.src[self.node_features_key],
            ]
        else:
            input_data_list = [
                x.dst[self.node_features_key],
                x.dst[self.node_hidden_state_key],
                x.src[self.node_features_key],
                x.src[self.node_hidden_state_key],
            ]
        #print("x.src[self.node_features_key] ",x.src[self.node_features_key][0:50])
        #print("x.src[self.node_features_key].shape ",x.src[self.node_features_key].shape)
        #print("x.data[self.edge_feature_key] ",x.data[self.edge_feature_key][0:50])
        #print("x.data[self.edge_feature_key].shape ",x.data[self.edge_feature_key].shape)
        #print("************************")

        assert x.data[self.edge_feature_key].dim() == 1
        input_data_list.append(x.data[self.edge_feature_key].unsqueeze(dim=1))

        for data in input_data_list:
            assert data.dim() == 2
            assert data.size(dim=0) == x.batch_size()

        input_data = torch.cat(input_data_list, dim=1)

        result = self.net(input_data)

        output = {self.edge_hidden_state_key: result}
        #print("output self.edge_hidden_state_key: output of edge node nn",output[self.edge_hidden_state_key].shape)

        return output


class NodeNetwork(nn.Module):
    def __init__(
        self,
        input_node_features_size: int,
        input_node_hidden_state_size: int,
        input_edge_hidden_state_size: int,
        output_node_hidden_state_size: int,
        dropout: float,
        first_layer: bool,
        node_features_key: str,
        node_hidden_state_key: str,
        edge_hidden_state_key: str,
    ) -> None:
        super(NodeNetwork, self).__init__()

        if output_node_hidden_state_size % 2 != 0:
            raise ValueError(
                "'output_node_hidden_state_size' has to be an even number because "
                "the output is a concatenation of two neural networks which "
                "should each have output_node_hidden_state_size/2 outputs. "
                f"Got: {output_node_hidden_state_size = }"
            )

        self.input_node_features_size = input_node_features_size
        self.input_node_hidden_state_size = input_node_hidden_state_size
        self.input_edge_hidden_state_size = input_edge_hidden_state_size
        self.output_node_hidden_state_size = output_node_hidden_state_size
        self.dropout = dropout
        self.first_layer = first_layer
        self.node_features_key = node_features_key
        self.node_hidden_state_key = node_hidden_state_key
        self.edge_hidden_state_key = edge_hidden_state_key

        # out_features for both nets is half of the desired
        # output features of the node update
        out_features_both_nets = int(self.output_node_hidden_state_size / 2)

        in_features_net_1 = (
            self.input_node_features_size + self.input_node_hidden_state_size
        )

        mid_features_net_1 = int(
            (
                self.input_node_features_size
                + self.input_node_hidden_state_size
                + out_features_both_nets
            )
            / 2
        )

        self.net_1 = nn.Sequential(
            nn.Linear(
                in_features=in_features_net_1,
                out_features=mid_features_net_1,
                bias=True,
            ),
            nn.Dropout(p=self.dropout),
            nn.ReLU(),
            nn.Linear(
                in_features=mid_features_net_1,
                out_features=out_features_both_nets,
                bias=True,
            ),
            nn.Tanh(),
        )

        self.net_2 = nn.Sequential(
            nn.Linear(
                in_features=self.input_edge_hidden_state_size,
                out_features=out_features_both_nets,
                bias=True,
            ),
            nn.Dropout(p=self.dropout),
            nn.ReLU(),
            nn.Linear(
                in_features=out_features_both_nets,
                out_features=out_features_both_nets,
                bias=True,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        #print("x in forward pass is ",x)
        #print("forward pass of node update called")
        if self.first_layer is True:
            input_data_1 = x.data[self.node_features_key]
        else:
            input_data_1 = torch.cat(
                [x.data[self.node_features_key], x.data[self.node_hidden_state_key]],
                dim=1,
            )

        result_1 = self.net_1(input_data_1)

        # message sum of incoming edges
        #print("x.mailbox[self.edge_hidden_state_key].shape nefore sum ",x.mailbox[self.edge_hidden_state_key].shape)
        #print("x.mailbox[self.edge_hidden_state_key] ",x.mailbox[self.edge_hidden_state_key])
        input_data_2 = torch.sum(x.mailbox[self.edge_hidden_state_key], dim=1)
        #print("x.mailbox[self.edge_hidden_state_key].shape after sum ",input_data_2.shape)
        #print("input_data_2 ",input_data_2)

        result_2 = self.net_2(input_data_2)
        #print("result_2 ",result_2.shape) #torch.Size([20, 256])
        #print("result_1 ",result_1.shape) #torch.Size([20, 256])
        result = torch.cat([result_1, result_2], dim=1)
        #print("result ",result[0]) #torch.Size([20, 512])
        result = result / torch.norm(result, p="fro", dim=1, keepdim=True)
        #print("result ",result[0])
        #>>> result                                                                                                                                                                                             tensor([[1, 2],                                                                                                                                                                                                [3, 4]])                                                                                                                                                                                       >>> torch.norm(result.float(), p="fro", dim=1, keepdim=True)                                                                                                                                           tensor([[2.2361],                                                                                                                                                                                              [5.0000]]) 
        #happens sqrt(across row)
        output = {self.node_hidden_state_key: result}

        return output


class GNNTorchModel(nn.Module):
    def __init__(
        self,
        n_node_features: int,
        edge_hidden_state_sizes: List[int],
        node_hidden_state_sizes: List[int],
        jet_efficiency_net_hidden_layers: List[int],
        n_classes: int,
        flavour_embedding_num_embeddings: Optional[int],
        flavour_embedding_dim: Optional[int],
        edge_network_dropout: float,
        node_network_dropout: float,
        jet_efficiency_net_dropout: float,
        node_features_key: str,
        node_hidden_state_key: str,
        edge_feature_key: str,
        edge_hidden_state_key: str,
        flavour_indices_key: str,
    ) -> None:
        super(GNNTorchModel, self).__init__()

        if len(edge_hidden_state_sizes) != len(node_hidden_state_sizes):
            raise ValueError(
                "Lists of hidden state sizes must have the same length. Got: "
                f"{len(edge_hidden_state_sizes) = }, "
                f"{len(node_hidden_state_sizes) = }"
            )

        if (flavour_embedding_num_embeddings is None) != (
            flavour_embedding_dim is None
        ):
            raise ValueError(
                "Parameters for flavour embedding have to either "
                "all be None or all have values. Got: "
                f"{flavour_embedding_num_embeddings = }, "
                f"{flavour_embedding_dim = }"
            )

        self.n_node_features = n_node_features
        self.edge_hidden_state_sizes = edge_hidden_state_sizes
        self.node_hidden_state_sizes = node_hidden_state_sizes
        self.jet_efficiency_net_hidden_layers = jet_efficiency_net_hidden_layers
        self.n_classes = n_classes
        self.flavour_embedding_num_embeddings = flavour_embedding_num_embeddings
        self.flavour_embedding_dim = flavour_embedding_dim
        self.edge_network_dropout = edge_network_dropout
        self.node_network_dropout = node_network_dropout
        self.jet_efficiency_net_dropout = jet_efficiency_net_dropout
        self.node_features_key = node_features_key
        self.node_hidden_state_key = node_hidden_state_key
        self.edge_feature_key = edge_feature_key
        self.edge_hidden_state_key = edge_hidden_state_key
        self.flavour_indices_key = flavour_indices_key

        self.node_predictions_key = "NODE_PREDICTION"

        if self.flavour_embedding_dim is not None:
            self.flavour_embedding = nn.Embedding(
                num_embeddings=self.flavour_embedding_num_embeddings,
                embedding_dim=self.flavour_embedding_dim,
            )
            self.total_node_features = self.n_node_features + self.flavour_embedding_dim
        else:
            self.flavour_embedding = None
            self.total_node_features = self.n_node_features

        self.edge_updates = nn.ModuleList()
        self.node_updates = nn.ModuleList()

        for i in range(len(self.edge_hidden_state_sizes)):
            # This GN block:
            ## uses a node hidden state of size
            if i == 0:
                # zero, as this is the first GN block and no node hidden state exists yet
                gn_block_input_node_hidden_state_size = 0
            else:
                # of the node hidden state of the previous GN block
                gn_block_input_node_hidden_state_size = self.node_hidden_state_sizes[
                    i - 1
                ]

            ## builds an edge representation of size in the EdgeNetwork
            ## and uses it as input in the NodeNetwork
            gn_block_output_edge_hidden_state_size = self.edge_hidden_state_sizes[i]

            ## builds a node hidden state size of
            gn_block_output_node_hidden_state_size = self.node_hidden_state_sizes[i]

            if i == 0:
                first_layer = True
            else:
                first_layer = False
             
            edge_network = EdgeNetwork(
                input_node_features_size=self.total_node_features,
                input_node_hidden_state_size=gn_block_input_node_hidden_state_size,
                output_edge_hidden_state_size=gn_block_output_edge_hidden_state_size,
                dropout=self.edge_network_dropout,
                first_layer=first_layer,
                node_features_key=self.node_features_key,
                node_hidden_state_key=self.node_hidden_state_key,
                edge_feature_key=self.edge_feature_key,
                edge_hidden_state_key=self.edge_hidden_state_key,
            )

            node_network = NodeNetwork(
                input_node_features_size=self.total_node_features,
                input_node_hidden_state_size=gn_block_input_node_hidden_state_size,
                input_edge_hidden_state_size=gn_block_output_edge_hidden_state_size,
                output_node_hidden_state_size=gn_block_output_node_hidden_state_size,
                dropout=self.node_network_dropout,
                first_layer=first_layer,
                node_features_key=self.node_features_key,
                node_hidden_state_key=self.node_hidden_state_key,
                edge_hidden_state_key=self.edge_hidden_state_key,
            )

            self.edge_updates.append(edge_network)
            self.node_updates.append(node_network)

        jet_efficiency_net_layers = []

        for i in range(len(self.jet_efficiency_net_hidden_layers)):
            if i == 0:  # first layer
                # input of node features, embedded flavour, and the
                # node hidden state from the last GN block
                in_features = (
                    self.total_node_features + self.node_hidden_state_sizes[-1]
                )
                out_features = self.jet_efficiency_net_hidden_layers[0]
            else:
                in_features = self.jet_efficiency_net_hidden_layers[i - 1]
                out_features = self.jet_efficiency_net_hidden_layers[i]
            jet_efficiency_net_layers.extend(
                [
                    nn.Linear(
                        in_features=in_features,
                        out_features=out_features,
                        bias=True,
                    ),
                    nn.Dropout(p=self.jet_efficiency_net_dropout),
                    nn.ReLU(),
                ]
            )

        jet_efficiency_net_layers.append(
            nn.Linear(
                in_features=self.jet_efficiency_net_hidden_layers[-1],
                out_features=self.n_classes,
                bias=True,
            )
        )

        self.jet_efficiency_net = nn.Sequential(*jet_efficiency_net_layers)

    def forward(self, g):
        if self.flavour_embedding is not None:
            # pass flavour indices through embedding and append it to the node features
            #print("g.ndata[self.flavour_indices_key].shape ",g.ndata[self.flavour_indices_key].shape)  #torch.Size([4635])
            #print("g.ndata[self.node_features_key].shape ",g.ndata[self.node_features_key].shape) #torch.Size([4635, 3])
            
            embedded_flavour = self.flavour_embedding(g.ndata[self.flavour_indices_key]) 
            g.ndata[self.node_features_key] = torch.cat(
                [
                    g.ndata[self.node_features_key],
                    embedded_flavour,
                ],
                dim=1,
            )
            #print("g.ndata[self.node_features_key].shape after torch.cat",g.ndata[self.node_features_key].shape)  #torch.Size([4635, 5])

        for edge_update, node_update in zip(self.edge_updates, self.node_updates):
            g.update_all(
                message_func=edge_update,
                reduce_func=node_update,
            )
            #print("calling update all")
#        print("g.ndata[self.node_hidden_state_key] before jet layers ",g.ndata[self.node_hidden_state_key])
        print("g.ndata[self.node_hidden_state_key].shape",g.ndata[self.node_hidden_state_key].shape) #torch.Size([2329, 512]) 

        jet_efficiency_net_input = torch.cat(
            [
                g.ndata[self.node_features_key],
                g.ndata[self.node_hidden_state_key],
            ],
            dim=1,
        )

        jet_efficiency_net_output = self.jet_efficiency_net(jet_efficiency_net_input)

        g.ndata[self.node_predictions_key] = jet_efficiency_net_output

        out = g.ndata[self.node_predictions_key]

        return out
