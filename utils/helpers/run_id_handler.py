from __future__ import annotations

import datetime as dt
import pathlib
import uuid
from typing import List, Optional

from loguru import logger


class RunIdHandler:
    possible_prefix_values = (
        "in_progress",
        "failed",
    )

    def __init__(self, run_id: str, prefix: Optional[str]) -> None:
        self._run_id = None
        self._prefix = None

        self.run_id = run_id
        self.prefix = prefix

    @property
    def run_id(self):
        return self._run_id

    @run_id.setter
    def run_id(self, value: str):
        if not isinstance(value, str):
            raise ValueError("Argument must be a str")
        minimum_length = 10
        if len(value) < minimum_length:
            raise ValueError(
                f"Argument must be a str with a length greater than {minimum_length}"
            )

        self._run_id = value

    @property
    def prefix(self):
        return self._prefix

    @prefix.setter
    def prefix(self, value: Optional[str]):
        if value is not None and not isinstance(value, str):
            raise ValueError("Argument must be None or a str")
        if value is not None and value not in self.possible_prefix_values:
            raise ValueError(
                f"Unsupported value for prefix: '{value}. "
                f"Possible values are: {self.possible_prefix_values}"
            )

        self._prefix = value

    def get_str(self):
        if self.prefix is None:
            return str(self.run_id)
        else:
            return f"{self.prefix}_{self.run_id}"

    @classmethod
    def new_run_id(cls, prefix: Optional[str], bootstrap: bool) -> RunIdHandler:
        run_id = cls.generate_run_id(bootstrap=bootstrap)
        inst = cls(run_id=run_id, prefix=prefix)
        return inst

    @staticmethod
    def generate_run_id(bootstrap: bool) -> str:
        run_id = (
            f"{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}__{str(uuid.uuid4())}"
        )

        if bootstrap is True:
            run_id = RunIdHandler.convert_to_bootstrap_run_id(run_id=run_id)

        return run_id

    @staticmethod
    def convert_to_bootstrap_run_id(run_id: str) -> str:
        return f"bootstrap_{run_id}"

    @staticmethod
    def is_bootstrap_run_id(run_id: str) -> bool:
        return run_id.startswith("bootstrap_")

    @classmethod
    def get_run_ids(cls, dir_path: pathlib.Path, only_latest: bool) -> List[str]:
        # TODO(critical): allow filtering on bootstrap or not
        if not dir_path.exists() or not dir_path.is_dir():
            return []
        else:
            run_ids = []
            for p in dir_path.iterdir():
                if p.is_dir() and not any(
                    p.name.startswith(prefix) for prefix in cls.possible_prefix_values
                ):
                    run_ids.append(p.name)

            run_ids = sorted(run_ids)

            logger.trace(f"Found {len(run_ids)} start_times in {dir_path}")

            if only_latest:
                return [run_ids[-1]]
            else:
                return run_ids
