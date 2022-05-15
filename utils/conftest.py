import copy

import pytest
from loguru import logger

import utils
from utils import Paths

logger.remove(None)  # remove all logging handlers


@pytest.fixture
def test_paths() -> Paths:
    paths = copy.deepcopy(utils.paths)
    paths.root_path = paths.root_path / "test_data"

    return paths
