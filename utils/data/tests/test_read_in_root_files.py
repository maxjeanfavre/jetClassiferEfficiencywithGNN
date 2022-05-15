from typing import List, Optional, Union

import pandas as pd
import pytest
import uproot.exceptions

from utils import Paths
from utils.data.read_in_root_files import (
    read_in_root_files_in_chunks,
    read_in_root_files_via_np,
    read_in_root_files_via_pd,
)
from utils.exceptions import TTreeBranchesRowLengthsMismatchError


@pytest.fixture
def test_root_files_dir_path(test_paths: Paths):
    return str(test_paths.root_path / "input") + "/"


@pytest.fixture
def key():
    return "Events;1"


@pytest.fixture
def filenames():
    return (
        "tree_1_sample.root",
        "tree_2_sample.root",
        "tree_3_sample.root",
        "tree_4_sample.root",
        "tree_5_sample.root",
    )


class TestReadInRootFilesInChunks:
    @pytest.mark.parametrize("library", ["np", "pd"])
    @pytest.mark.parametrize("chunk_size", [1, 2, 5])
    @pytest.mark.parametrize("num_workers", [1, 2])
    @pytest.mark.parametrize(
        "expressions",
        [
            "Jet_pt",
            ["Jet_eta", "Jet_phi"],
        ],
    )
    @pytest.mark.parametrize("file_limit", [1, 3, 5])
    def test_read_in_root_files_in_chunks(
        self,
        test_root_files_dir_path,
        filenames,
        key,
        file_limit: Optional[int],
        expressions: Optional[Union[str, List[str]]],
        num_workers: int,
        chunk_size: int,
        library: str,
    ):
        res = read_in_root_files_in_chunks(
            path=test_root_files_dir_path,
            filenames=filenames,
            file_limit=file_limit,
            key=key,
            expressions=expressions,
            num_workers=num_workers,
            chunk_size=chunk_size,
            library=library,
        )

        file_paths = [test_root_files_dir_path + filename for filename in filenames]
        files = {path: key for path in file_paths[:file_limit]}

        if library == "np":
            comp_res = read_in_root_files_via_np(
                files=files,
                expressions=expressions,
                num_workers=num_workers,
            )
        elif library == "pd":
            comp_res = read_in_root_files_via_pd(
                files=files,
                expressions=expressions,
                num_workers=num_workers,
            )
        else:
            assert False

        pd.testing.assert_frame_equal(
            left=res,
            right=comp_res,
        )


@pytest.mark.parametrize("library", ["np", "pd"])
class TestReadInRootFiles:
    # uproot error when trying to read in a non-existent branch
    @pytest.mark.parametrize(
        "expressions", ["thiswontbeinthere42", ["loremipsumwhy", "naphhg"]]
    )
    def test_read_in_root_files_key_in_file_error(
        self,
        test_root_files_dir_path,
        filenames,
        key,
        library: str,
        expressions: Union[str, List[str]],
    ):
        num_workers = 1
        file_paths = [test_root_files_dir_path + filename for filename in filenames]
        files = {path: key for path in file_paths[:1]}

        with pytest.raises(uproot.exceptions.KeyInFileError) as exc_info:
            if library == "np":
                read_in_root_files_via_np(
                    files=files,
                    expressions=expressions,
                    num_workers=num_workers,
                )
            elif library == "pd":
                read_in_root_files_via_pd(
                    files=files,
                    expressions=expressions,
                    num_workers=num_workers,
                )
            else:
                assert False
            if isinstance(expressions, str):
                assert expressions in exc_info.value
            if isinstance(expressions, list):
                assert expressions[0] in exc_info.value

    @pytest.mark.parametrize(
        "expressions", [["Jet_Pt", "FatJet_Pt"], ["Jet_eta", "FatJet_phi"]]
    )
    def test_read_in_root_files_branches_lengths_mismatch_error(
        self,
        test_root_files_dir_path,
        filenames,
        key,
        library: str,
        expressions: List[str],
    ):
        num_workers = 1
        file_paths = [test_root_files_dir_path + filename for filename in filenames]
        files = {path: key for path in file_paths[:1]}

        with pytest.raises(TTreeBranchesRowLengthsMismatchError) as exc_info:
            if library == "np":
                read_in_root_files_via_np(
                    files=files,
                    expressions=expressions,
                    num_workers=num_workers,
                )
            elif library == "pd":
                read_in_root_files_via_pd(
                    files=files,
                    expressions=expressions,
                    num_workers=num_workers,
                )
            else:
                assert False
            assert expressions[0] in exc_info.value


class TestEqualityOfPandasAndNumpyReadIn:
    @pytest.mark.parametrize("num_workers", [1, 2])
    @pytest.mark.parametrize(
        "expressions",
        [
            "Jet_pt",
            ["Jet_eta", "Jet_phi"],
        ],
    )
    @pytest.mark.parametrize("file_limit", [1, 3, 5])
    def test_equality(
        self,
        test_root_files_dir_path,
        filenames,
        key,
        file_limit: int,
        expressions: Optional[Union[str, List[str]]],
        num_workers: int,
    ):
        file_paths = [test_root_files_dir_path + filename for filename in filenames]
        files = {path: key for path in file_paths[:file_limit]}

        pandas_read_in = read_in_root_files_via_pd(
            files=files,
            expressions=expressions,
            num_workers=num_workers,
        )

        numpy_read_in = read_in_root_files_via_np(
            files=files,
            expressions=expressions,
            num_workers=num_workers,
        )

        pd.testing.assert_frame_equal(
            left=pandas_read_in,
            right=numpy_read_in,
        )
