from __future__ import annotations

import pathlib
import pickle
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from utils.configs.working_points_set import WorkingPointsSetConfig
from utils.data.jet_events_dataset import JetEventsDataset
from utils.efficiency_map.histogram import Histogram
from utils.helpers.bayesian_efficiency import (
    compute_mean_bayesian_efficiency,
    compute_mode_bayesian_efficiency,
    compute_variance_bayesian_efficiency,
)


class BinnedEfficiencyMaps:
    def __init__(
        self,
        histograms: Dict[str, Dict[Union[str, Tuple], Dict[str, Histogram]]],
        separation_cols: Tuple[str, ...],
        working_points_set_config: WorkingPointsSetConfig,
    ) -> None:
        self._histograms = None
        self._separation_cols = None
        self._working_points_set_config = None

        self.separation_cols = separation_cols
        self.working_points_set_config = working_points_set_config
        self.histograms = histograms

    @property
    def histograms(self):
        return self._histograms

    @histograms.setter
    def histograms(self, value: Dict[str, Dict[Any, Dict[str, Histogram]]]):
        if not isinstance(value, dict):
            raise ValueError(f"'value' has to be a dict. Got type: {type(value)}")
        for k0, v0 in value.items():
            if not isinstance(k0, str) or not isinstance(v0, dict):
                raise ValueError(
                    "'value' has to be a dict[str, dict]. "
                    f"One entry had key {k0} and value {v0}"
                )
            for k1, v1 in v0.items():
                if not isinstance(v1, dict):
                    raise ValueError(
                        "1st level values of 'value' have to be "
                        "a dict[Any, dict]]. "
                        f"Got: key: {k1}, value: {v1}"
                    )
                for k2, v2 in v1.items():
                    if not isinstance(k2, str) or not isinstance(v2, Histogram):
                        raise ValueError(
                            "2nd level values of 'value' have "
                            "to be a Dict[str, Histogram]]. "
                            f"Got: key: {k2}, value: {v2}"
                        )

        if self.separation_cols is not None:
            for working_point_name, working_point_histograms in value.items():
                for separator_value in working_point_histograms.keys():
                    if len(self.separation_cols) != 1:
                        if len(separator_value) != len(self.separation_cols):
                            raise ValueError(
                                "Found separator_value inconsistent "
                                "with the separation_cols. "
                                f"separator_value: {separator_value}, "
                                f"separation_cols: {self.separation_cols}"
                            )

        if self.working_points_set_config is not None:
            working_point_names = list(value.keys())
            working_point_names_expected = [
                working_point_config.name
                for working_point_config in self.working_points_set_config.working_points
            ]
            if working_point_names != working_point_names_expected:
                raise ValueError(
                    "working_point_names are inconsistent with "
                    "existing working_points_set_config. "
                    f"working_point_names: {working_point_names}, "
                    f"working_point_names_expected: {working_point_names_expected}"
                )

        for working_point_name, working_point_histograms in value.items():
            for (
                separator_values,
                separated_histograms,
            ) in working_point_histograms.items():
                if set(separated_histograms.keys()) != {
                    "all",
                    "passed",
                    "eff_mode",
                    "eff_mean",
                    "eff_variance",
                }:
                    raise ValueError(
                        f"Missing or unexpected histograms in "
                        f"working_point {working_point_name} "
                        f"separator_values {separator_values}. "
                        f"Got: {set(separated_histograms.keys())}"
                    )

        self._histograms = value

    @property
    def separation_cols(self):
        return self._separation_cols

    @separation_cols.setter
    def separation_cols(self, value: Tuple[str, ...]):
        if not isinstance(value, tuple):
            raise ValueError(
                "Value for must be a tuple with str entries. "
                f"Got an element of type: {type(value)}"
            )
        if any(not isinstance(i, str) for i in value):
            raise ValueError(
                "Value must be a tuple with str entries. "
                f"The entries had types {[type(i) for i in value]}"
            )
        if self.histograms is not None:
            for _, working_point_histograms in self.histograms.items():
                separator_values = list(working_point_histograms.keys())
                if len(value) == 1:
                    # edge case where values in jds.df[separation_col] is
                    # already e.g. a tuple is not allowed then right now
                    pass
                else:
                    if any(
                        len(separator_value) != len(value)
                        for separator_value in separator_values
                    ):
                        raise ValueError("Value was inconsistent with self.histograms")

        self._separation_cols = value

    @property
    def working_points_set_config(self):
        return self._working_points_set_config

    @working_points_set_config.setter
    def working_points_set_config(self, value: WorkingPointsSetConfig):
        if not isinstance(value, WorkingPointsSetConfig):
            raise ValueError(
                "Value must be an instance of 'WorkingPointsSetConfig. "
                f"Got type: {type(value)}"
            )
        if self.histograms is not None:
            if list(self.histograms.keys()) != [wp.name for wp in value.working_points]:
                raise ValueError(
                    "Value was inconsistent with the working "
                    "point names in self.histograms"
                )

        self._working_points_set_config = value

    @classmethod
    def create_efficiency_maps(
        cls,
        jds: JetEventsDataset,
        bins: Dict[str, np.ndarray],
        separation_cols: Tuple[str, ...],
        working_points_set_config: WorkingPointsSetConfig,
    ) -> BinnedEfficiencyMaps:
        histograms = dict()

        for working_point_config in working_points_set_config.working_points:
            working_point_histograms = dict()
            for separator_values, group_df in jds.df.groupby(
                by=list(separation_cols), sort=False
            ):
                hist_all = Histogram.from_df_and_bins(
                    df=group_df,
                    bins=bins,
                )

                discriminator_mask = group_df.eval(working_point_config.expression)

                hist_passed = Histogram.from_df_and_bins(
                    df=group_df[discriminator_mask],
                    bins=bins,
                )

                hist_eff_mode = compute_mode_bayesian_efficiency(
                    k=hist_passed, n=hist_all
                )

                hist_eff_mean = compute_mean_bayesian_efficiency(
                    k=hist_passed, n=hist_all
                )

                hist_eff_variance = compute_variance_bayesian_efficiency(
                    k=hist_passed, n=hist_all
                )

                working_point_histograms[separator_values] = {
                    "all": hist_all,
                    "passed": hist_passed,
                    "eff_mode": hist_eff_mode,
                    "eff_mean": hist_eff_mean,
                    "eff_variance": hist_eff_variance,
                }
            histograms[working_point_config.name] = working_point_histograms

        bem = cls(
            histograms=histograms,
            separation_cols=separation_cols,
            working_points_set_config=working_points_set_config,
        )

        return bem

    def save(self, path: pathlib.Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: pathlib.Path) -> BinnedEfficiencyMaps:
        with open(path, "rb") as f:
            bem = pickle.load(f)
        return bem

    def compute_efficiency(
        self, jds: JetEventsDataset, working_point_name: str
    ) -> Tuple[pd.Series, pd.Series]:
        eff = pd.Series(index=jds.df.index, dtype="float64")
        eff_var = pd.Series(index=jds.df.index, dtype="float64")
        for separator_values, group_df in jds.df.groupby(
            by=list(self.separation_cols), sort=False
        ):
            hist_eff = self.histograms[working_point_name][separator_values]["eff_mode"]
            eff_res = hist_eff.get_bin_entry(df=group_df)

            eff.loc[
                group_df.index  # TODO(low): maybe slow because of the index, also below
            ] = eff_res

            hist_eff_var = self.histograms[working_point_name][separator_values][
                "eff_variance"
            ]
            eff_var_res = hist_eff_var.get_bin_entry(df=group_df)

            eff_var.loc[group_df.index] = eff_var_res

        for name, s in [["eff", eff], ["eff_var", eff_var]]:
            n_nan_values = s.isna().sum()
            if n_nan_values != 0:
                logger.warning(
                    f"Got {n_nan_values} nan values in {name} "
                    f"at working point: {working_point_name}"
                )

            n_negative_values = np.sum(s < 0)
            if n_negative_values != 0:
                logger.critical(
                    f"Got {n_negative_values} values < 0 in {name} "
                    f"at working point: {working_point_name}"
                )

            n_zero_values = np.sum(s == 0)
            if n_zero_values != 0:
                logger.warning(
                    f"Got {n_zero_values} values = 0 in {name} "
                    f"at working point: {working_point_name}"
                )

            n_above_1_values = np.sum(s > 1)
            if n_above_1_values != 0:
                logger.critical(
                    f"Got {n_above_1_values} values > 1 in {name} "
                    f"at working point: {working_point_name}"
                )

            # res = np.where(
            #     res < 10e-15, 10e-15, res
            # )  # replace values below threshold (can't use 0 because of possible division with these values)
            # res = np.where(
            #     res > 1 - 10e-15, 1 - 10e-15, res
            # )  # replace values above threshold

        return eff, eff_var

    def project(self, projection_variables) -> BinnedEfficiencyMaps:
        # project 'all' and 'passed' histogram along projection
        # variables and recalculate eff from that
        new_histograms = dict()
        for working_point_name, working_point_histograms in self.histograms.items():
            new_working_point_histograms = dict()
            for (
                separator_values,
                separated_histograms,
            ) in working_point_histograms.items():
                hist_all = separated_histograms["all"]
                hist_all_projected = hist_all.project(
                    projection_variables=projection_variables
                )

                hist_passed = separated_histograms["passed"]
                hist_passed_projected = hist_passed.project(
                    projection_variables=projection_variables
                )

                hist_eff_projected = hist_passed_projected / hist_all_projected

                hist_eff_mode_projected = compute_mode_bayesian_efficiency(
                    k=hist_passed_projected, n=hist_all_projected
                )

                hist_eff_mean_projected = compute_mean_bayesian_efficiency(
                    k=hist_passed_projected, n=hist_all_projected
                )

                hist_eff_variance_projected = compute_variance_bayesian_efficiency(
                    k=hist_passed_projected, n=hist_all_projected
                )

                new_working_point_histograms[separator_values] = {
                    "all": hist_all_projected,
                    "passed": hist_passed_projected,
                    "eff_mode": hist_eff_mode_projected,
                    "eff_mean": hist_eff_mean_projected,
                    "eff_variance": hist_eff_variance_projected,
                }

            new_histograms[working_point_name] = new_working_point_histograms

        bem_projected = BinnedEfficiencyMaps(
            histograms=new_histograms,
            separation_cols=self.separation_cols,
            working_points_set_config=self.working_points_set_config,
        )

        return bem_projected

    def without_under_over_flow(self) -> BinnedEfficiencyMaps:
        new_histograms = {}
        for working_point_name, working_point_histograms in self.histograms.items():
            new_working_point_histograms = dict()
            for (
                separator_values,
                separated_histograms,
            ) in working_point_histograms.items():
                hist_all = separated_histograms["all"]
                hist_all_without_under_over_flow = hist_all.without_under_over_flow()

                hist_passed = separated_histograms["passed"]
                hist_passed_without_under_over_flow = (
                    hist_passed.without_under_over_flow()
                )

                hist_eff_mode = separated_histograms["eff_mode"]
                hist_eff_mode_without_under_over_flow = (
                    hist_eff_mode.without_under_over_flow()
                )

                assert (
                    hist_eff_mode_without_under_over_flow
                    == compute_mode_bayesian_efficiency(
                        k=hist_passed_without_under_over_flow,
                        n=hist_all_without_under_over_flow,
                    )
                )

                hist_eff_mean = separated_histograms["eff_mean"]
                hist_eff_mean_without_under_over_flow = (
                    hist_eff_mean.without_under_over_flow()
                )

                assert (
                    hist_eff_mean_without_under_over_flow
                    == compute_mean_bayesian_efficiency(
                        k=hist_passed_without_under_over_flow,
                        n=hist_all_without_under_over_flow,
                    )
                )

                hist_eff_variance = separated_histograms["eff_mean"]
                hist_eff_variance_without_under_over_flow = (
                    hist_eff_variance.without_under_over_flow()
                )

                assert (
                    hist_eff_variance_without_under_over_flow
                    == compute_variance_bayesian_efficiency(
                        k=hist_passed_without_under_over_flow,
                        n=hist_all_without_under_over_flow,
                    )
                )

                new_working_point_histograms[separator_values] = {
                    "all": hist_all_without_under_over_flow,
                    "passed": hist_passed_without_under_over_flow,
                    "eff_mode": hist_eff_mode_without_under_over_flow,
                    "eff_mean": hist_eff_mean_without_under_over_flow,
                    "eff_variance": hist_eff_variance_without_under_over_flow,
                }
            new_histograms[working_point_name] = new_working_point_histograms

        bem_without_under_over_flow = BinnedEfficiencyMaps(
            histograms=new_histograms,
            separation_cols=self.separation_cols,
            working_points_set_config=self.working_points_set_config,
        )

        return bem_without_under_over_flow
