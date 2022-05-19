from typing import Dict, List

import numpy as np
import sklearn.metrics
from loguru import logger

from utils.data.jet_events_dataset import JetEventsDataset

import torch
from torch import nn


def compute_predictions_loss(
    jds: JetEventsDataset, eff_pred_cols: List[str], comparison_col: str
) -> Dict:
    y_comp = jds.df[comparison_col].to_numpy()

    data = {}

    for eff_col in eff_pred_cols:
        y = jds.df[eff_col].to_numpy()

        data[eff_col] = compute_predictions_loss_raw(y_pred=y, y_true=y_comp)

    return data


def compute_predictions_loss_raw(y_pred: np.ndarray, y_true: np.ndarray) -> Dict:
    y_pred_mask = ~np.isnan(y_pred)
    y_true_mask = ~np.isnan(y_true)
    logger.debug(f"y_pred_mask mean: {np.mean(y_pred_mask)}")
    logger.debug(f"y_true_mask mean: {np.mean(y_true_mask)}")

    y_pred = y_pred[np.logical_and(y_pred_mask, y_true_mask)]
    y_true = y_true[np.logical_and(y_pred_mask, y_true_mask)]

    rmse = compute_rmse(y_1=y_pred, y_2=y_true)
    bce = sklearn.metrics.log_loss(y_true=y_true, y_pred=y_pred)

    with torch.no_grad():
        rmse_torch = torch.sqrt(
            nn.MSELoss()(torch.FloatTensor(y_pred), torch.FloatTensor(y_true))
        ).item()
        bce_torch = nn.BCELoss()(
            torch.FloatTensor(y_pred), torch.FloatTensor(y_true)
        ).item()

    res = {
        "rmse": rmse,
        "rmse_torch": rmse_torch,
        "bce": bce,
        "bce_torch": bce_torch,
    }

    return res


def compute_rmse(y_1: np.ndarray, y_2: np.ndarray):
    rmse = np.sqrt(np.mean(np.square(y_1 - y_2)))

    return rmse
