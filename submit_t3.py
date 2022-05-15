import datetime as dt
import subprocess
from typing import Dict, List, Optional

import utils
from main import get_parsers
from utils.helpers.modules.package_names import get_package_names

python_executable = "/work/phummler/miniconda3/envs/gnn_efficiency/bin/python"


def get_bash_run_id_command(bootstrap: bool):
    return f"""RUNID=$(/work/phummler/miniconda3/envs/gnn_efficiency/bin/python -c "from utils.helpers.run_id_handler import RunIdHandler; print(RunIdHandler.generate_run_id(bootstrap={bootstrap}));" >&1)"""


def get_commands(
    task: str,
    dataset: str,
    dataset_handling: Optional[str],
    working_points_set: Optional[str],
    model: Optional[str],
    bootstrap: bool,
    run_id: Optional[str],
    prediction_dataset_handling: Optional[str],
    evaluation_model_selection: Optional[str],
    evaluation_data_manipulation_list: Optional[List[str]],
    save_predictions_after_training_prediction_dataset_handling: Optional[List[str]],
) -> List[str]:
    main_script = "main.py"
    cmd = f"{python_executable} {main_script}"
    cmd += f" {task}"
    cmd += f" --dataset={dataset}"
    if dataset_handling is not None:
        cmd += f" --dataset_handling={dataset_handling}"
    if working_points_set is not None:
        cmd += f" --working_points_set={working_points_set}"
    if model is not None:
        cmd += f" --model={model}"
    if bootstrap is True:
        cmd += " --bootstrap"
    if run_id is not None:
        cmd += f" --run_id={run_id}"
    if prediction_dataset_handling is not None:
        cmd += f" --prediction_dataset_handling={prediction_dataset_handling}"
    if evaluation_model_selection is not None:
        cmd += f" --evaluation_model_selection={evaluation_model_selection}"
    if evaluation_data_manipulation_list is not None:
        cmd += f" --evaluation_data_manipulation {' '.join(evaluation_data_manipulation_list)}"

    if save_predictions_after_training_prediction_dataset_handling is not None:
        cmd += " --run_id=$RUNID"
        commands = [
            get_bash_run_id_command(bootstrap=bootstrap),
            cmd,
            *[
                cmd
                for e in save_predictions_after_training_prediction_dataset_handling
                for cmd in get_commands(
                    task="save_predictions",
                    dataset=dataset,
                    dataset_handling=dataset_handling,
                    working_points_set=working_points_set,
                    model=model,
                    bootstrap=False,  # False as bootstrap is not an argument for parser_save_predictions
                    run_id="$RUNID",
                    prediction_dataset_handling=e,
                    evaluation_model_selection=None,  # None, as it is not an argument for parser_save_predictions
                    evaluation_data_manipulation_list=None,  # None, as it is not an argument for parser_save_predictions
                    save_predictions_after_training_prediction_dataset_handling=None,
                )
            ],
        ]
    else:
        commands = [cmd]

    return commands


def get_slurm_kwargs(
    task: str,
    dataset: str,
    dataset_handling: Optional[str],
    working_points_set: Optional[str],
    model: Optional[str],
    bootstrap: Optional[bool],
    run_id: Optional[str],
    prediction_dataset_handling: Optional[str],
    evaluation_model_selection: Optional[str],
    evaluation_data_manipulation_list: Optional[List[str]],
):
    job_name = task

    is_test_dataset = dataset.endswith("_test")

    if task == "extract":
        job_name += f"_{dataset}"
        cpus_per_task = 8
        n_gpus = 0
        if is_test_dataset:
            time_hours = 0.5
            mem_gb = 10
        else:
            time_hours = 2
            mem_gb = 80
    elif task == "train":
        job_name += (
            f"_{dataset}"
            f"_{dataset_handling}"
            f"_{working_points_set}"
            f"_{model}"
            f"_{bootstrap}"
        )
        if model.startswith("gnn"):
            cpus_per_task = 2
            n_gpus = 1
            if is_test_dataset:
                time_hours = 1
                mem_gb = 10
            else:
                time_hours = 12
                mem_gb = 48
        elif model.startswith("eff_map"):
            cpus_per_task = 4
            n_gpus = 0
            if is_test_dataset:
                time_hours = 0.5
                mem_gb = 10
            else:
                time_hours = 1
                mem_gb = 60
        elif model.startswith("direct_tagging"):
            cpus_per_task = 4
            n_gpus = 0
            if is_test_dataset:
                time_hours = 0.5
                mem_gb = 10
            else:
                time_hours = 0.5
                mem_gb = 60
        else:
            raise ValueError(f"Unknown model: {model}")
    elif task == "save_predictions":
        job_name += (
            f"_{dataset}_"
            f"{dataset_handling}_"
            f"{working_points_set}_"
            f"{model}_"
            f"{run_id}_"
            f"{prediction_dataset_handling}"
        )
        if model.startswith("gnn"):
            cpus_per_task = 2
            n_gpus = 1
            if is_test_dataset:
                time_hours = 0.5
                mem_gb = 10
            else:
                time_hours = 4
                mem_gb = 48
        elif model.startswith("eff_map"):
            cpus_per_task = 4
            n_gpus = 0
            if is_test_dataset:
                time_hours = 0.5
                mem_gb = 10
            else:
                time_hours = 1
                mem_gb = 60
        elif model.startswith("direct_tagging"):
            cpus_per_task = 4
            n_gpus = 0
            if is_test_dataset:
                time_hours = 0.5
                mem_gb = 10
            else:
                time_hours = 0.5
                mem_gb = 60
        else:
            raise ValueError(f"Unknown model: {model}")
    elif task == "evaluate":
        job_name += (
            f"_{dataset}_"
            f"{dataset_handling}_"
            f"{working_points_set}_"
            f"{prediction_dataset_handling}_"
            f"{evaluation_model_selection}_"
            f"{'__'.join(evaluation_data_manipulation_list)}"
        )
        cpus_per_task = 2
        n_gpus = 0
        if is_test_dataset:
            time_hours = 0.5
            mem_gb = 10
        else:
            time_hours = 1
            mem_gb = 40

    else:
        raise ValueError(f"Unknown task: {task}")

    gres = None
    if n_gpus > 0:
        if time_hours <= 1:
            # partition = "qgpu"  # I think I can only one concurrent job on qgpu
            partition = "gpu"
        elif time_hours <= 7 * 24:
            partition = "gpu"
        else:
            raise ValueError(
                f"'time_hours' can't be longer than one week ({7*24} hours). "
                f"Got {time_hours} hours"
            )
        account = "gpu_gres"
        gres = f"gpu:{n_gpus}"
    else:
        if time_hours <= 1:
            partition = "short"
        elif time_hours <= 12:
            partition = "standard"
        elif time_hours <= 7 * 24:
            partition = "long"
        else:
            raise ValueError(
                f"'time_hours' can't be longer than one week ({7*24} hours). "
                f"Got {time_hours} hours"
            )
        account = "t3"

    assert (
        time_hours < 24
    )  # otherwise the str from dt.timedelta is not in a format that works with slurm
    time_ = str(
        dt.timedelta(seconds=int(time_hours * 3600))
    )  # seconds instead of hours to make sure seconds is int so I don't get a
    # str like from str(dt.timedelta(hours=0.011))

    nodes = "1"
    ntasks = "1"
    cpus_per_task = str(int(cpus_per_task))
    mem = f"{int(mem_gb)}G"
    output = "slurm_%j-%x.out"
    error = "slurm_%j-%x.err"

    res = {
        "partition": partition,
        "account": account,
        "time": time_,
        "nodes": nodes,
        "ntasks": ntasks,
        "cpus-per-task": cpus_per_task,
        "mem": mem,
        "job-name": job_name,
        "output": output,
        "error": error,
    }

    if gres is not None:
        res["gres"] = gres

    return res


def submit_to_slurm(commands: List[str], slurm_kwargs: Dict[str, str]):
    try:
        subprocess.run(["sinfo", "--version"])
    except FileNotFoundError as file_not_found_error:
        raise OSError(
            "Slurm seems to be not available. Make sure to run this on T3"
        ) from file_not_found_error

    s = "sbatch <<'EOT'" + "\n"
    s += "#!/bin/bash" + "\n"
    s += "\n"

    for key, value in slurm_kwargs.items():
        s += f"#SBATCH --{key}={value}" + "\n"

    s += "\n"

    s += 'echo "START: $(date)"' + "\n"

    s += "cd /work/phummler/code/" + "\n"

    for command in commands:
        s += command + "\n"

    s += 'echo "END: $(date)"' + "\n"

    s += "EOT"

    print(s)

    subprocess.run(s, shell=True)


def submit_to_t3(
    task: str,
    dataset: str,
    dataset_handling: Optional[str],
    working_points_set: Optional[str],
    model: Optional[str],
    bootstrap: Optional[bool],
    run_id: Optional[str],
    prediction_dataset_handling: Optional[str],
    evaluation_model_selection: Optional[str],
    evaluation_data_manipulation_list: Optional[List[str]],
    save_predictions_after_training_prediction_dataset_handling: Optional[List[str]],
):
    commands = get_commands(
        task=task,
        dataset=dataset,
        dataset_handling=dataset_handling,
        working_points_set=working_points_set,
        model=model,
        bootstrap=bootstrap,
        run_id=run_id,
        prediction_dataset_handling=prediction_dataset_handling,
        evaluation_model_selection=evaluation_model_selection,
        evaluation_data_manipulation_list=evaluation_data_manipulation_list,
        save_predictions_after_training_prediction_dataset_handling=save_predictions_after_training_prediction_dataset_handling,
    )

    slurm_kwargs = get_slurm_kwargs(
        task=task,
        dataset=dataset,
        dataset_handling=dataset_handling,
        working_points_set=working_points_set,
        model=model,
        bootstrap=bootstrap,
        run_id=run_id,
        prediction_dataset_handling=prediction_dataset_handling,
        evaluation_model_selection=evaluation_model_selection,
        evaluation_data_manipulation_list=evaluation_data_manipulation_list,
    )

    submit_to_slurm(
        commands=commands,
        slurm_kwargs=slurm_kwargs,
    )


def main():
    (
        parser,
        parser_train,
        parser_save_predictions,
        parser_evaluate,
        parser_extract,
    ) = get_parsers()

    parser_train.add_argument(
        "--save_predictions_after_training_prediction_dataset_handling",
        nargs="*",
        choices=get_package_names(
            dir_path=utils.paths.config(
                config_type="prediction_dataset_handling",
                config_name="foo",
                mkdir=False,
            ).parent
        ),
        help="names of the prediction dataset handling config",
    )

    args = parser.parse_args()

    task = args.task

    dataset = args.dataset

    try:
        save_predictions_after_training_prediction_dataset_handling = (
            args.save_predictions_after_training_prediction_dataset_handling
        )
    except AttributeError:
        save_predictions_after_training_prediction_dataset_handling = None

    try:
        dataset_handling = args.dataset_handling
    except AttributeError:
        dataset_handling = None

    try:
        working_points_set = args.working_points_set
    except AttributeError:
        working_points_set = None

    try:
        model = args.model
    except AttributeError:
        model = None

    try:
        bootstrap = args.bootstrap
    except AttributeError:
        bootstrap = None

    try:
        run_id = args.run_id
    except AttributeError:
        run_id = None

    try:
        prediction_dataset_handling = args.prediction_dataset_handling
    except AttributeError:
        prediction_dataset_handling = None

    try:
        evaluation_model_selection = args.evaluation_model_selection
    except AttributeError:
        evaluation_model_selection = None

    try:
        evaluation_data_manipulation_list = args.evaluation_data_manipulation
    except AttributeError:
        evaluation_data_manipulation_list = None

    submit_to_t3(
        task=task,
        dataset=dataset,
        dataset_handling=dataset_handling,
        working_points_set=working_points_set,
        model=model,
        bootstrap=bootstrap,
        run_id=run_id,
        prediction_dataset_handling=prediction_dataset_handling,
        evaluation_model_selection=evaluation_model_selection,
        evaluation_data_manipulation_list=evaluation_data_manipulation_list,
        save_predictions_after_training_prediction_dataset_handling=save_predictions_after_training_prediction_dataset_handling,
    )


if __name__ == "__main__":
    main()
