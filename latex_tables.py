import json

import numpy as np

import utils
from utils.helpers.run_id_handler import RunIdHandler

# TODO(medium): config.display_name
nicify = {
    "btagWP_Loose_DeepCSV": "Loose",
    "btagWP_Medium_DeepCSV": "Medium",
    "btagWP_Tight_DeepCSV": "Tight",
    "QCD_Pt_300_470_MuEnrichedPt5": "QCD",
    "QCD_Pt_300_470_MuEnrichedPt5_test": "QCD (test)",
    "TTTo2L2Nu": r"$\mathrm{t\bar{t}}$",
    "TTTo2L2Nu_test": r"$\mathrm{t\bar{t}}$ (test)",
    "chi_squared": r"$\chi^2$",
    "rmse": "RMSE",
}


def histogram_metrics_table(
    metric,
    models,
    evaluation_data_histograms,
    dataset,
    working_point,
    include_label: bool,
):
    table = ""
    table += r"\begin{table}" + "\n"
    table += r"\centering" + "\n"
    table += r"\scriptsize" + "\n"
    table += r"\setlength\tabcolsep{2pt}" + "\n"
    table += (
        r"\caption{Histogram evaluation metrics for the "
        + nicify.get(dataset, dataset)
        + " dataset using the "
        + nicify.get(working_point, working_point)
        + " working point."
        + (
            " "
            "They measure the distance between the histograms from the efficiency "
            "weighting techniques to the histogram from direct tagging."
        )
        # + (" " "The best value in each row is shown in bold.")
        + "}"
        + "\n"
    )

    table += r"\begin{tabular}{l " + len(models) * "S" + "}" + "\n"
    table += r"\toprule" + "\n"

    table += (
        r"&\multicolumn{"
        + str(len(models))
        + "}{c}{"
        + nicify.get(metric, metric)
        + "}"
        + r"\\"
        + "\n"
    )

    table += "\cmidrule(lr){" + str(2) + "-" + str(2 + len(models) - 1) + "}" + "\n"

    table += "Histogram "
    for model in models:
        table += "& {" + model.replace("_", " ") + "} "
    table += r"\\" + "\n"

    table += r"\midrule" + "\n"

    for histogram_name, histogram_data in evaluation_data_histograms.items():
        lowest_value = np.inf
        for model in models:
            value = histogram_data[model][metric]
            if value < lowest_value:
                lowest_value = value
            del value
        table += histogram_name + " "
        for model in models:
            table += "& "
            value = histogram_data[model][metric]
            if value == np.inf:
                table += r"$\infty$" + " "
            else:
                if value == lowest_value:
                    table += r"\bfseries "
                table += f"{value:.2f}" + " "
        table += r"\\" + "\n"

    table += r"\bottomrule" + "\n"
    table += (
        r"\multicolumn{" + str(len(models) + 1) + r"}{l}{Note: "
        r"The best value in each row is shown in bold.}" + "\n"
    )
    table += r"\end{tabular}" + "\n"
    if include_label is True:
        table += (
            r"\label{tbl:histogram_metrics_"
            + dataset
            + "_"
            + working_point
            + "}"
            + "\n"
        )
    table += r"\end{table}"

    return table


def predictions_loss_table(
    metrics,
    models,
    evaluation_data_predictions_loss,
    dataset,
    working_point,
    include_label: bool,
):
    table = ""
    table += r"\begin{table}" + "\n"
    table += r"\centering" + "\n"
    table += r"\scriptsize" + "\n"
    table += r"\setlength\tabcolsep{2pt}" + "\n"
    table += (
        r"\caption{Evaluation metrics of the efficiency predictions for the "
        + nicify.get(dataset, dataset)
        + " dataset using the "
        + nicify.get(working_point, working_point)
        + " working point."
        + (
            " "
            "The root-mean-square error~(RMSE) and the binary cross entropy~(BCE) are "
            "computed using the pseudo-efficiency from direct tagging as true values."
        )
        # + (" " "The best value in each column is shown in bold.")
        + "}"
        + "\n"
    )

    table += r"\begin{tabular}{l " + len(metrics) * "S" + "}" + "\n"
    table += r"\toprule" + "\n"

    table += "Model "
    for metric in metrics:
        table += "& {" + metric + "} "
    table += r"\\" + "\n"

    table += r"\midrule" + "\n"

    lowest_values = {metric: np.inf for metric in metrics}
    for model in models:
        model_data = evaluation_data_predictions_loss[model]
        for metric in metrics:
            value = model_data[metric]
            if value is not None and value < lowest_values[metric]:
                lowest_values[metric] = value
    for model in models:
        model_data = evaluation_data_predictions_loss[model]
        table += model + " "
        for metric in metrics:
            table += "& "
            value = model_data[metric]
            if value == np.inf:
                table += r"$\infty$" + " "
            else:
                if value == lowest_values[metric]:
                    table += r"\bfseries "
                if value is not None:
                    table += f"{value:.3f}" + " "
                else:
                    table += "none "
        table += r"\\" + "\n"

    table += r"\bottomrule" + "\n"
    table += (
        r"\multicolumn{" + str(len(metrics) + 1) + r"}{l}{Note: "
        r"The best value in each column is shown in bold.}" + "\n"
    )
    table += r"\end{tabular}" + "\n"
    if include_label is True:
        table += (
            r"\label{tbl:prediction_loss_" + dataset + "_" + working_point + "}" + "\n"
        )
    table += r"\end{table}"

    return table


def main():
    from configs.dataset.QCD_Pt_300_470_MuEnrichedPt5 import (
        dataset_config as QCD_Pt_300_470_MuEnrichedPt5_dataset_config,
    )
    from configs.dataset.TTTo2L2Nu import dataset_config as TTTo2L2Nu_dataset_config
    from configs.dataset_handling.standard_dataset_handling import (
        dataset_handling_config as standard_dataset_handling_config,
    )
    from configs.working_points_set.standard_working_points_set import (
        working_points_set_config as standard_working_points_set_config,
    )
    from configs.prediction_dataset_handling.no_changes import (
        prediction_dataset_handling_config as no_changes_prediction_dataset_handling_config,
    )
    from configs.evaluation_model_selection.selection_1_mean import (
        evaluation_model_selection_config as selection_1_mean_evaluation_model_selection_config,
    )
    from configs.evaluation_model_selection.selection_1_median import (
        evaluation_model_selection_config as selection_1_median_evaluation_model_selection_config,
    )
    from configs.evaluation_model_selection.selection_1_mean_and_median import (
        evaluation_model_selection_config as selection_1_mean_and_median_evaluation_model_selection_config,
    )
    from configs.evaluation_model_selection.individual_gnn_and_median import (
        evaluation_model_selection_config as individual_gnn_and_median_evaluation_model_selection_config,
    )
    from configs.evaluation_model_selection.individual_gnn_variables_1_and_median import (
        evaluation_model_selection_config as individual_gnn_variables_1_and_median_evaluation_model_selection_config,
    )
    from configs.evaluation_data_manipulation.no_changes import (
        evaluation_data_manipulation_config as no_changes_evaluation_data_manipulation_config,
    )
    from configs.evaluation_data_manipulation.pt_eta_cut import (
        evaluation_data_manipulation_config as pt_eta_cut_evaluation_data_manipulation_config,
    )
    from configs.evaluation_data_manipulation.events_valid_btagDeepB import (
        evaluation_data_manipulation_config as events_valid_btagDeepB_evaluation_data_manipulation_config,
    )
    from configs.evaluation_data_manipulation.events_valid_btagDeepB_pt_eta_cut import (
        evaluation_data_manipulation_config as events_valid_btagDeepB_pt_eta_cut_evaluation_data_manipulation_config,
    )

    for dataset_config in [
        TTTo2L2Nu_dataset_config,
        QCD_Pt_300_470_MuEnrichedPt5_dataset_config,
    ]:
        for dataset_handling_config in [standard_dataset_handling_config]:
            for working_points_set_config in [standard_working_points_set_config]:
                for prediction_dataset_handling_config in [
                    no_changes_prediction_dataset_handling_config
                ]:
                    for evaluation_model_selection_config in [
                        selection_1_mean_evaluation_model_selection_config,
                        selection_1_median_evaluation_model_selection_config,
                        selection_1_mean_and_median_evaluation_model_selection_config,
                        individual_gnn_and_median_evaluation_model_selection_config,
                        individual_gnn_variables_1_and_median_evaluation_model_selection_config,
                    ]:
                        for evaluation_data_manipulation_config in [
                            no_changes_evaluation_data_manipulation_config,
                            pt_eta_cut_evaluation_data_manipulation_config,
                            events_valid_btagDeepB_evaluation_data_manipulation_config,
                            events_valid_btagDeepB_pt_eta_cut_evaluation_data_manipulation_config,
                        ]:
                            evaluation_name = (
                                f"{prediction_dataset_handling_config.name}_"
                                f"{evaluation_model_selection_config.name}_"
                                f"{evaluation_data_manipulation_config.name}"
                            )
                            run_ids = RunIdHandler.get_run_ids(
                                dir_path=utils.paths.evaluation_files_dir(
                                    dataset_name=dataset_config.name,
                                    dataset_handling_name=dataset_handling_config.name,
                                    working_points_set_name=working_points_set_config.name,
                                    evaluation_name=evaluation_name,
                                    run_id="foo",
                                    working_point_name="foo",
                                    mkdir=False,
                                ).parent.parent,
                                only_latest=False,
                            )
                            for run_id in run_ids:
                                for (
                                    working_point_config
                                ) in working_points_set_config.working_points:
                                    evaluation_dir_path = utils.paths.evaluation_files_dir(
                                        dataset_name=dataset_config.name,
                                        dataset_handling_name=dataset_handling_config.name,
                                        working_points_set_name=working_points_set_config.name,
                                        evaluation_name=evaluation_name,
                                        run_id=run_id,
                                        working_point_name=working_point_config.name,
                                        mkdir=False,
                                    )
                                    with open(
                                        evaluation_dir_path / "evaluation_data.json",
                                        "r",
                                    ) as f:
                                        evaluation_data = json.load(fp=f)

                                    evaluation_data_histograms = {}
                                    for evaluation_data_key in [
                                        "jet_variable_histograms",
                                        "leading_subleading_histograms",
                                    ]:
                                        for histogram_key in sorted(
                                            list(
                                                evaluation_data[
                                                    evaluation_data_key
                                                ].keys()
                                            )
                                        ):
                                            assert (
                                                histogram_key
                                                not in evaluation_data_histograms
                                            )
                                            evaluation_data_histograms[
                                                histogram_key
                                            ] = evaluation_data[evaluation_data_key][
                                                histogram_key
                                            ]

                                    sample_histogram_key = next(
                                        iter(evaluation_data_histograms)
                                    )
                                    models = list(
                                        evaluation_data_histograms[
                                            sample_histogram_key
                                        ].keys()
                                    )
                                    metrics = list(
                                        evaluation_data_histograms[
                                            sample_histogram_key
                                        ][models[0]].keys()
                                    )
                                    # checking that there are the same models and metrics for all histograms
                                    for (
                                        histogram_data
                                    ) in evaluation_data_histograms.values():
                                        assert set(models) == set(histogram_data.keys())
                                        for model_data in histogram_data.values():
                                            assert set(metrics) == set(
                                                model_data.keys()
                                            )

                                    for metric in metrics:
                                        for include_label in [True, False]:
                                            table = histogram_metrics_table(
                                                metric=metric,
                                                models=models,
                                                evaluation_data_histograms=evaluation_data_histograms,
                                                dataset=dataset_config.name,
                                                working_point=working_point_config.name,
                                                include_label=include_label,
                                            )
                                            if include_label is True:
                                                filename = (
                                                    f"table_histograms_{metric}.tex"
                                                )
                                            else:
                                                filename = f"table_histograms_{metric}_no_label.tex"
                                            with open(
                                                evaluation_dir_path / filename,
                                                "w",
                                            ) as f:
                                                f.write(table)

                                    evaluation_data_predictions_loss = evaluation_data[
                                        "predictions_loss"
                                    ]
                                    metrics = list(
                                        evaluation_data_predictions_loss[
                                            models[0]
                                        ].keys()
                                    )
                                    metrics = [m for m in metrics if "torch" not in m]

                                    for include_label in [True, False]:
                                        table = predictions_loss_table(
                                            metrics=metrics,
                                            models=models,
                                            evaluation_data_predictions_loss=evaluation_data_predictions_loss,
                                            dataset=dataset_config.name,
                                            working_point=working_point_config.name,
                                            include_label=include_label,
                                        )
                                        if include_label is True:
                                            filename = "table_prediction_loss.tex"
                                        else:
                                            filename = (
                                                "table_prediction_loss_no_label.tex"
                                            )
                                        with open(
                                            evaluation_dir_path / filename,
                                            "w",
                                        ) as f:
                                            f.write(table)


if __name__ == "__main__":
    main()
