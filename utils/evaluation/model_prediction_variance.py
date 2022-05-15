from utils.configs.model import ModelConfig
from utils.data.jet_events_dataset import JetEventsDataset


# TODO(high): do this
def model_prediction_variance(
    jds: JetEventsDataset,
    model_config: ModelConfig,
):
    for flavour, flavour_df in jds.df.groupby("Jet_hadronFlavour", sort=True):
        flavour_model_run_predictions_df = flavour_df.filter(
            regex=f"res_{model_config.name}"
        )
        flavour_pred_mean = flavour_model_run_predictions_df.mean(axis=1)
        a = flavour_model_run_predictions_df.sub(flavour_pred_mean, axis=0).div(
            flavour_pred_mean, axis=0
        )
        print(flavour)
        b = a.describe()
        print(b)

    # model_run_predictions_df = jds.df.filter(
    #     regex=f"res_{model_config.name}"
    # )  # TODO(low): the res_ part is not set in stone right now

    # leading_subleading_eff_product_df = model_run_predictions_df.loc[(slice(None), slice(0, 1)), :].groupby(level=0, sort=False).prod(
    #     min_count=2
    # )
    # leading_subleading_eff_product_mean = leading_subleading_eff_product_df.mean(axis=1)

    # a = leading_subleading_eff_product_df.sub(leading_subleading_eff_product_mean, axis=0).div(leading_subleading_eff_product_mean, axis=0)
    # print(a.describe())
    # b = a.describe()
    # pred_std_dev = model_run_predictions_df.std(axis=1)
