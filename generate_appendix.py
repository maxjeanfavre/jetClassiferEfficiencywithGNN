import utils
from latex_tables import nicify
from utils.configs.dataset import DatasetConfig
from utils.configs.working_points_set import WorkingPointsSetConfig


def create_appendix(
    dataset_config: DatasetConfig,
    working_points_set_config: WorkingPointsSetConfig,
    base_path: str,
):
    s = ""
    s += (
        r"\chapter{"
        + nicify.get(dataset_config.name, dataset_config.name)
        + " dataset"
        + "}"
        + "\n"
    )
    s += r"\label{appendix_ch:" + dataset_config.name + "}" + 2 * "\n"
    for working_point_config in working_points_set_config.working_points:
        s += (
            r"\section{"
            + nicify.get(working_point_config.name, working_point_config.name)
            + " Working Point in the "
            + nicify.get(dataset_config.name, dataset_config.name)
            + " dataset"
            + "}"
            + 2 * "\n"
        )
        # predictions loss table
        s += (
            r"\input{"
            + base_path
            + dataset_config.name
            + "/evaluation/"
            + working_point_config.name
            + "/"
            + "table_prediction_loss_no_label.tex"
            + "}"
            + 2 * "\n"
        )

        # histograms table
        # rmse
        s += (
            r"\input{"
            + base_path
            + dataset_config.name
            + "/evaluation/"
            + working_point_config.name
            + "/"
            + "table_histograms_rmse_no_label.tex"
            + "}"
            + 2 * "\n"
        )
        # chi squared
        s += (
            r"\input{"
            + base_path
            + dataset_config.name
            + "/evaluation/"
            + working_point_config.name
            + "/"
            + "table_histograms_chi_squared_no_label.tex"
            + "}"
            + 2 * "\n"
        )

        histogram_caption_standard_end = ""
        histogram_caption_standard_end += (
            f" The {nicify.get(working_point_config.name, working_point_config.name)} "
            f"working point is shown."
        )
        histogram_caption_standard_end += (
            " The top pad shows the distribution resulting from direct tagging and "
            "different efficiency weighting methods as indicated by the colour "
            "legend. The middle pad shows the ratio of the bin values of the "
            "efficiency methods and direct tagging. The bottom pad shows the "
            "ratio of the statistical uncertainty of the bin values of the "
            "efficiency methods and direct tagging."
        )
        # histogram plots
        histograms = []
        # jet variable histograms
        for histogram_name, variable_name_nice in [
            ["Jet_eta", r"Jet pseudorapidity $\eta$"],
            ["Jet_phi", r"Jet azimuthal angle $\phi$"],
            ["Jet_Pt", r"Jet transverse momentum $p_\text{T}$"],
            ["Jet_area", "Jet catchment area"],
            ["Jet_mass", "Jet mass"],
        ]:
            for flavour_selection in ["inclusive", "5"]:
                histogram_filename = (
                    f"jet_variable_histogram_{histogram_name}_{flavour_selection}.pdf"
                )

                caption = f"{variable_name_nice} "
                if flavour_selection == "5":
                    caption += r"of b~jets"
                elif flavour_selection == "inclusive":
                    caption += "for all flavours"
                else:
                    raise ValueError(f"Unknown flavour selection: {flavour_selection}")
                caption += (
                    f" in the "
                    f"{nicify.get(dataset_config.name, dataset_config.name)}"
                    f" dataset."
                )
                caption += histogram_caption_standard_end

                histograms.append(
                    [histogram_name, histogram_filename, caption, flavour_selection]
                )
        # leading subleading histograms
        for histogram_name, variable_name_nice in [
            ["invariant_mass", r"Invariant mass $M_{1,2}$"],
            ["delta_r", r"Angular distance $\Delta R$"],
        ]:
            for flavour_selection in ["inclusive", "5_5"]:
                histogram_filename = (
                    f"leading_subleading_{histogram_name}_{flavour_selection}.pdf"
                )

                caption = (
                    f"{variable_name_nice} of the leading and "
                    f"sub-leading jet in the "
                    f"{nicify.get(dataset_config.name, dataset_config.name)} "
                    f"dataset."
                )
                if flavour_selection == "5_5":
                    caption += (
                        " The histogram is restricted to those events where "
                        "the leading and sub-leading jets were both b~jets."
                    )
                elif flavour_selection == "inclusive":
                    pass
                else:
                    raise ValueError(f"Unknown flavour selection: {flavour_selection}")
                caption += histogram_caption_standard_end

                histograms.append(
                    [histogram_name, histogram_filename, caption, flavour_selection]
                )

        for (
            histogram_name,
            histogram_filename,
            caption,
            flavour_selection,
        ) in histograms:
            s += r"\begin{figure}" + "\n"
            s += r"\centering" + "\n"
            s += (
                r"\includegraphics[width=\textwidth]{"
                + base_path
                + dataset_config.name
                + "/evaluation/"
                + working_point_config.name
                + "/"
                + histogram_filename
                + "}"
                + "\n"
            )
            s += r"\caption{" + caption + "}" + "\n"
            s += (
                r"\label{appendix_fig:"
                + dataset_config.name
                + "_"
                + working_point_config.name
                + "_"
                + histogram_name
                + "_"
                + flavour_selection
                + "}"
                + "\n"
            )
            s += r"\end{figure}" + "\n"
            s += "\n"

        s += r"\FloatBarrier" + "\n"
        s += "\n"
    print(s)

    with open(
        utils.paths.dataset_outputs_dir(dataset_name=dataset_config.name, mkdir=True)
        / "appendix.tex",
        "w",
    ) as f:
        f.write(s)


def main():
    from configs.dataset.QCD_Pt_300_470_MuEnrichedPt5 import (
        dataset_config as QCD_Pt_300_470_MuEnrichedPt5_dataset_config,
    )
    from configs.dataset.TTT
2L2Nu import dataset_config as TTTo2L2Nu_dataset_config
    from configs.working_points_set.standard_working_points_set import (
        working_points_set_config as standard_working_points_set_config,
    )

    base_path = "data/"
    for dataset_config in [
        TTTo2L2Nu_dataset_config,
        QCD_Pt_300_470_MuEnrichedPt5_dataset_config,
    ]:
        create_appendix(
            dataset_config=dataset_config,
            working_points_set_config=standard_working_points_set_config,
            base_path=base_path,
        )


if __name__ == "__main__":
    main()
