import math
import pathlib
import time
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import uproot
from loguru import logger

from utils.data.dataframe_format import (
    get_idx_from_event_n_jets,
    reconstruct_event_n_jets_from_groupby_zeroth_level_values,
)
from utils.exceptions import TTreeBranchesRowLengthsMismatchError, UnexpectedError
from utils.helpers.basics.chunks import chunks


def read_in_root_files_in_chunks(
    path: str,
    filenames: Tuple[str, ...],
    file_limit: Optional[int] = None,
    key: str = "Events;1",
    expressions: Optional[Union[str, List[str]]] = None,
    num_workers: int = 10,
    chunk_size: int = 10,
    library: str = "np",
):
    logger.info("Starting reading in root files in chunks")
    root_file_paths = [path + filename for filename in filenames]

    logger.debug(f"Got {len(root_file_paths)} files")
    if file_limit is not None:
        if file_limit < len(root_file_paths):
            logger.debug(f"Limiting to the first {file_limit} files")
        root_file_paths = root_file_paths[:file_limit]

    if expressions is not None:
        if isinstance(expressions, str):
            expressions = [expressions]
        expressions = list(set(expressions))

    s = time.time()
    chunk_res_list = []
    event_n_jets_list = []
    total_chunks = math.ceil(len(root_file_paths) / chunk_size)
    for i, root_file_paths_chunk in enumerate(
        chunks(lst=root_file_paths, n=chunk_size)
    ):
        logger.trace(f"Reading in {i + 1}/{total_chunks} chunk")
        chunk_files = {path: key for path in root_file_paths_chunk}
        if library == "np":
            chunk_res = read_in_root_files_via_np(
                files=chunk_files,
                expressions=expressions,
                num_workers=num_workers,
            )
        elif library == "pd":
            chunk_res = read_in_root_files_via_pd(
                files=chunk_files,
                expressions=expressions,
                num_workers=num_workers,
            )
        else:
            raise ValueError(
                f"Unsupported value for 'library': {library}. "
                f"Supported values are: 'np' and 'pd'"
            )
        chunk_res_list.append(chunk_res)
        event_n_jets_chunk = reconstruct_event_n_jets_from_groupby_zeroth_level_values(
            df=chunk_res
        )
        event_n_jets_list.append(event_n_jets_chunk)
    e = time.time()
    logger.trace(f"Reading in took {e - s:.2f} seconds")

    logger.trace("Generating result DataFrame")
    res = pd.concat(chunk_res_list, axis=0)

    event_n_jets = np.concatenate(event_n_jets_list)
    idx = get_idx_from_event_n_jets(event_n_jets=event_n_jets)
    res.index = idx
    logger.trace("Done generating result DataFrame")

    logger.debug("Done reading in root files in chunks")

    return res


def read_in_root_files_via_pd(
    files: Dict[Union[pathlib.Path, str], str],
    expressions: Optional[Union[str, List[str]]],
    num_workers: int,
) -> pd.DataFrame:
    if expressions is not None:
        if isinstance(expressions, str):
            expressions = [expressions]
        expressions = list(set(expressions))

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        res = uproot.concatenate(
            files=files,
            expressions=expressions,
            library="pd",
            allow_missing=False,
            num_workers=num_workers,
        )

    if not isinstance(res, pd.DataFrame):
        if isinstance(res, tuple) and all([isinstance(i, pd.DataFrame) for i in res]):
            raise TTreeBranchesRowLengthsMismatchError(
                "Implied number of jets per event do not match "
                f"when reading in expressions {expressions} from files {files}. "
                "The result was divided in these groups of branches: "
                f"{[i.columns for i in res]}"
            )
        else:
            raise UnexpectedError(
                f"Result was neither pd.DataFrame nor tuple of pd.DataFrame "
                f"when reading in expressions {expressions} from files {files}"
            )

    return res


def read_in_root_files_via_np(
    files: Dict[Union[pathlib.Path, str], str],
    expressions: Optional[Union[str, List[str]]],
    num_workers: int,
) -> pd.DataFrame:
    if expressions is not None:
        if isinstance(expressions, str):
            expressions = [expressions]
        expressions = list(set(expressions))

    data = uproot.concatenate(
        files=files,
        expressions=expressions,
        library="np",
        allow_missing=False,
        num_workers=num_workers,
    )
    n_events_per_branch = {
        branch_name: len(branch_values) for branch_name, branch_values in data.items()
    }
    if len(set(n_events_per_branch.values())) != 1:
        raise ValueError(
            f"Different number of events in the branches: {n_events_per_branch} "
            f"when reading in expressions {expressions} from files {files}"
        )

    event_n_jets_from_jagged_branches = {
        k: np.array([len(i) for i in v])
        for k, v in data.items()
        if isinstance(v[0], np.ndarray)
    }
    first_key, *other_keys = event_n_jets_from_jagged_branches
    if not all(
        [
            np.array_equal(
                a1=event_n_jets_from_jagged_branches[first_key],
                a2=event_n_jets_from_jagged_branches[k],
            )
            for k in other_keys
        ]
    ):
        groups = []
        for (
            branch_name,
            branch_event_n_jets,
        ) in event_n_jets_from_jagged_branches.items():
            matching_group_found = False
            for group in groups:  # loop will be empty for the first branch
                if np.array_equal(a1=group["event_n_jets"], a2=branch_event_n_jets):
                    # sanity check to make sure only one group matches
                    assert matching_group_found is False
                    group["branch_names"].append(branch_name)
                    matching_group_found = True
            if matching_group_found is False:  # no group matches --> new group
                groups.append(
                    {"branch_names": [branch_name], "event_n_jets": branch_event_n_jets}
                )

        raise TTreeBranchesRowLengthsMismatchError(
            "Implied number of jets per event do not match "
            f"when reading in expressions {expressions} from files {files}. "
            "Found these groups of ragged branches with "
            "different number of entries per event: "
            f"{groups}"
        )

    event_n_jets = event_n_jets_from_jagged_branches[first_key]
    res = pd.DataFrame()
    # for branch_name, branch_values in data.items():
    for branch_name in list(data.keys()):
        branch_values = data[branch_name]
        if isinstance(branch_values[0], np.ndarray):
            res[branch_name] = np.concatenate(branch_values)
        else:
            res[branch_name] = np.repeat(branch_values, event_n_jets)
        del branch_values
        del data[branch_name]

    idx = get_idx_from_event_n_jets(event_n_jets=event_n_jets)
    res.index = idx

    return res
