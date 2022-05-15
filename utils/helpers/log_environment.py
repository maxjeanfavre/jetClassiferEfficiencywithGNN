import os

from loguru import logger


def log_environment():
    logger.debug(f"'os.getcwd()': {os.getcwd()}")
    environment_variables = [
        "PWD",
        "HOME",
        "USER",
        "SLURM_JOB_ID",
        "HOSTNAME",
        "CUDA_VISIBLE_DEVICES",
    ]
    for var in environment_variables:
        logger.debug(f"{var}: {os.environ.get(var)}")
