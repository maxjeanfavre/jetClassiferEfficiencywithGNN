import pathlib
import sys

from loguru import logger


def set_up_logging_sinks(dir_path: pathlib.Path, base_filename: str):
    assert dir_path.is_dir()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS!UTC}</green> | "
        "{elapsed} | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    try:
        logger.remove(None)  # removes all existing handlers
    except ValueError as e:
        logger.critical(str(e))

    logger.add(
        sink=sys.stdout,
        level="TRACE",
        format=log_format,
        colorize=True,
    )  # logging to sys.stdout instead of sys.stderr to keep .err files clean as a check

    logger.add(
        sink=dir_path / f"{base_filename}.log",
        level="TRACE",
        format=log_format,
        colorize=False,
        mode="w",
    )

    logger.add(
        sink=dir_path / f"{base_filename}_color.log",
        level="TRACE",
        format=log_format,
        colorize=True,
        mode="w",
    )
