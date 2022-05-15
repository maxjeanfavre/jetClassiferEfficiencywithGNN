from __future__ import annotations

import time
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import sklearn.model_selection
from loguru import logger

from utils.configs.dataset import DatasetConfig
from utils.data.dataframe_format import (
    check_df,
    get_idx_from_event_n_jets,
    reconstruct_event_n_jets_from_groupby_zeroth_level_values,
)
from utils.data.extraction.read_in_extract import read_in_extract
from utils.data.manipulation.data_manipulator import DataManipulator
from utils.data.read_in_root_files import read_in_root_files_in_chunks
from utils.exceptions import UnexpectedError
from utils.helpers.index_converter import IndexConverter


class JetEventsDataset:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = None

        self._n_events = None
        self._n_jets = None
        self._event_n_jets = None

        self.df = df

    @classmethod
    def read_in(
        cls,
        dataset_config: DatasetConfig,
        branches: Optional[Union[str, Tuple[str, ...]]] = None,
        run_id: Optional[str] = None,
    ):
        df = read_in_extract(
            dataset_config=dataset_config,
            branches=branches,
            run_id=run_id,
            return_all=False,
        )

        inst = cls(df=df)

        return inst

    @classmethod
    def from_root_file(
        cls,
        path: str,
        filenames: Tuple[str, ...],
        file_limit: Optional[int] = None,
        key: str = "Events;1",
        branches: Optional[
            Union[str, List[str]]
        ] = None,  # The relevant branches of the ROOT TTree.
        num_workers: int = 10,
        chunk_size: int = 10,
        library: str = "np",
    ) -> JetEventsDataset:
        df = read_in_root_files_in_chunks(
            path=path,
            filenames=filenames,
            file_limit=file_limit,
            key=key,
            expressions=branches,
            num_workers=num_workers,
            chunk_size=chunk_size,
            library=library,
        )

        inst = cls(df=df)
        return inst

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Argument must be a pd.DataFrame")

        logger.trace("Checking df")
        s = time.time()
        check_df(df=df)
        e = time.time()
        logger.trace(f"Check df took {e - s:.2f} seconds")

        self._df = df

        self._n_events = len(df.index.unique(level=0))
        self._n_jets = len(df)
        self._event_n_jets = reconstruct_event_n_jets_from_groupby_zeroth_level_values(
            df=df
        )

    @property
    def n_events(self):
        return self._n_events

    @property
    def n_jets(self):
        return self._n_jets

    @property
    def event_n_jets(self):
        return self._event_n_jets

    def manipulate(
        self, data_manipulators: Tuple[DataManipulator, ...], mode: str
    ) -> None:
        if len(data_manipulators) == 0:
            logger.debug("No data_manipulators to run")
        else:
            n_events_before = self.n_events
            n_jets_before = self.n_jets

            did_run = False
            for data_manipulator in data_manipulators:
                if mode in data_manipulator.active_modes:
                    data_manipulator.manipulate_inplace(jds=self)
                    did_run = True

            if not did_run:
                logger.warning(
                    f"No data_manipulators were run with selected mode: {mode}"
                )

            n_events_after = self.n_events
            n_jets_after = self.n_jets

            logger.debug(
                "After data manipulators, remaining: "
                f"{n_events_after} / {n_events_before} "
                f"({n_events_after / n_events_before * 100:.2f} %) of events, "
                f"{n_jets_after} / {n_jets_before} "
                f"({n_jets_after / n_jets_before * 100:.2f} %) of jets"
            )

    def split_data(
        self,
        train_size,
        test_size,
        return_train: Optional[bool] = False,
        return_test: Optional[bool] = False,
        random_state: Optional[int] = None,
        copy: bool = False,
    ):
        logger.debug(
            "Starting split_data with "
            f"train_size: {train_size}, test_size: {test_size}, "
            f"return_train: {return_train}, return_test: {return_test}, "
            f"random_state: {random_state}"
        )
        if not return_train and not return_test:
            raise ValueError(
                "At least one of 'return_train' and 'return_test' has to be True. "
                f"Got: return_train: {return_train}, return_test: {return_test}"
            )
        if train_size + test_size > 1:
            raise ValueError(
                "train_size + test_size has to be <= 1. "
                f"Got: train_size: {train_size}, test_size: {test_size}"
            )
        elif train_size == 1 or test_size == 1:
            if return_train and return_test:
                raise ValueError(
                    "If either train_size or test_size is 1, "
                    "only one of return_train and return_test can be True. "
                    f"Got: return_train: {return_train}, return_test: {return_test}"
                )
            assert return_train or return_test
            if copy:
                df = self.df.copy(deep=True)
                inst = JetEventsDataset(df=df)
                return inst
            else:
                logger.warning(
                    "Returning the original instance. Might lead to unexpected behavior"
                )
                return self
        else:
            # adapted from https://gist.github.com/shaypal5/3e34e85bd89d65d4ac118daa9a42b174
            logger.trace("Generating train test split of index")
            idx_train, idx_test = sklearn.model_selection.train_test_split(
                self.df.index.get_level_values(level=0).unique(),
                test_size=test_size,
                train_size=train_size,
                random_state=random_state,
                shuffle=True,
            )
            if return_train:
                logger.trace("Getting event_boolean_mask")
                idx_train = idx_train.to_numpy()
                idx_train = np.sort(idx_train)
                event_boolean_mask_train = (
                    IndexConverter.get_event_boolean_mask_from_event_index(
                        event_index=idx_train,
                        n_events=self.n_events,
                    )
                )

                logger.trace("Getting jet_boolean_mask")
                jet_boolean_mask_train = (
                    IndexConverter.get_jet_boolean_mask_from_event_boolean_mask(
                        event_boolean_mask=event_boolean_mask_train,
                        event_n_jets=self.event_n_jets,
                    )
                )
                logger.trace("Getting train_df with loc")
                train_df = self.df.loc[jet_boolean_mask_train]

                if copy:
                    train_df = train_df.copy()

                logger.trace("Getting event_n_jets")
                # have to use event_boolean_mask_train as this is used to get train_df
                # in the past idx_train wasn't sorted and therefore could not be used
                # to get event_n_jets_train because the order of events wouldn't match
                event_n_jets_train = self.event_n_jets[event_boolean_mask_train]

                logger.trace("Getting new index")
                train_new_idx = get_idx_from_event_n_jets(
                    event_n_jets=event_n_jets_train
                )

                logger.trace("Setting new index")
                train_df.index = train_new_idx

                logger.trace("Creating inst")
                inst_train = JetEventsDataset(df=train_df)

            if return_test:
                logger.trace("Getting event_boolean_mask")
                idx_test = idx_test.to_numpy()
                idx_test = np.sort(idx_test)
                event_boolean_mask_test = (
                    IndexConverter.get_event_boolean_mask_from_event_index(
                        event_index=idx_test,
                        n_events=self.n_events,
                    )
                )

                logger.trace("Getting jet_boolean_mask")
                jet_boolean_mask_test = (
                    IndexConverter.get_jet_boolean_mask_from_event_boolean_mask(
                        event_boolean_mask=event_boolean_mask_test,
                        event_n_jets=self.event_n_jets,
                    )
                )
                logger.trace("Getting test_df with loc")
                test_df = self.df.loc[jet_boolean_mask_test]

                if copy:
                    test_df = test_df.copy()

                logger.trace("Getting event_n_jets")
                # have to use event_boolean_mask_test as this is used to get test_df
                # in the past idx_test wasn't sorted and therefore could not be used
                # to get event_n_jets_test because the order of events wouldn't match
                event_n_jets_test = self.event_n_jets[event_boolean_mask_test]

                logger.trace("Getting new index")
                test_new_idx = get_idx_from_event_n_jets(event_n_jets=event_n_jets_test)

                logger.trace("Setting new index")
                test_df.index = test_new_idx

                logger.trace("Creating inst")
                inst_test = JetEventsDataset(df=test_df)

            logger.debug("Done with split_data")
            if return_train and return_test:
                return inst_train, inst_test
            elif return_train and not return_test:  # explicit here, unnecessary I know
                return inst_train
            elif not return_train and return_test:
                return inst_test
            else:
                raise UnexpectedError("You shouldn't reach this")
        # TODO(test): test whether the deep copy works in the train and
        #  test instances such that they are not effected by operations
        #  on the parent dataframe

    def get_bootstrap_sample(self, sample_size, return_bootstrap_idxs: bool = False):
        logger.debug(f"Starting 'get_bootstrap_sample' with sample_size: {sample_size}")

        if sample_size <= 0:
            raise ValueError("sample_size has to be strictly larger than 0")

        logger.trace("Generating index of bootstrap_selection")
        bootstrap_selection = sklearn.utils.resample(
            self.df.index.get_level_values(level=0).unique(),
            replace=True,
            n_samples=sample_size,
        )

        logger.trace(
            "Bootstrap fraction of selected events: "
            f"{len(set(bootstrap_selection))} / {self.n_events} "
            f"({len(set(bootstrap_selection)) / self.n_events * 100:.2f} %) of events"
        )

        counts = np.bincount(bootstrap_selection)
        idxs = []
        while (counts > 0).any():
            idxs.append(np.where(counts > 0)[0])
            counts -= 1

        dfs_to_concat = []
        for idx in idxs:
            logger.trace("Getting event_boolean_mask")
            event_boolean_mask = IndexConverter.get_event_boolean_mask_from_event_index(
                event_index=idx,
                n_events=self.n_events,
            )

            logger.trace("Getting jet_boolean_mask")
            jet_boolean_mask = (
                IndexConverter.get_jet_boolean_mask_from_event_boolean_mask(
                    event_boolean_mask=event_boolean_mask,
                    event_n_jets=self.event_n_jets,
                )
            )

            dfs_to_concat.append(self.df.loc[jet_boolean_mask])

        df = pd.concat(
            objs=dfs_to_concat,
            sort=False,
        )

        # can use groupby here as the level 0 index is unique because idxs is
        # constructed such that each sub-array only contains unique values
        event_n_jets_sub_dfs = [
            df.groupby(level=0, sort=False).size().to_numpy() for df in dfs_to_concat
        ]
        event_n_jets = np.concatenate(event_n_jets_sub_dfs)

        idx = get_idx_from_event_n_jets(event_n_jets=event_n_jets)
        df.index = idx

        inst = JetEventsDataset(df=df)

        if return_bootstrap_idxs:
            return inst, idxs
        else:
            return inst

    def __eq__(self, other):
        """Overrides the default implementation for equality."""
        if isinstance(other, JetEventsDataset):
            return (
                self.df.equals(other.df)
                and np.array_equal(a1=self.event_n_jets, a2=other.event_n_jets)
                and self.n_events == other.n_events
                and self.n_jets == other.n_jets
            )
        return NotImplemented
