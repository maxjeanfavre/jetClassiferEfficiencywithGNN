b-tagging efficiency with GNN

# Preparations
- set up conda environment `conda env create --file environment.yml`
  - or alternatively for an M1 mac `conda env create --file environment_no_cuda_m1.yml`
- `conda activate ...`
- Run some tests `pytest`
- Run some checks `python run_checks.py`

# Configs
There are different configuration files to change settings. They are selected through command-line options.

# Run
The `main.py` script is the entrypoint for these tasks:
- extract: Read in selected branches from all root files of a dataset and save them as one file. (Reading in the extraction later is way faster than reading in root files.)
- train: Train a model.
- save_predictions: Save predictions of a trained model. (Never used manually but only in automated job submissions - see below).
- evaluate: Run the evaluation.
The `submit_t3.py` can be used to submit jobs on PSI T3.

# Notes
- Provided enough storage is available, it makes sense to run the trainings with the `--save_predictions_after_training_prediction_dataset_handling` option. This will produce predictions after the training and saves them. Otherwise one would have to produce predictions during the evaluation which would take too much time (as (1) it would be done during each evaluation instead of once after the training; and (2) parallelization: it would produce predictions in the single evaluation job instead of in each training job)
