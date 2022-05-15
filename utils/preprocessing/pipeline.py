from __future__ import annotations

import pathlib
import pickle
from typing import Dict, List, Optional, Union

import pandas as pd

from utils.exceptions import NotFittedError
from utils.preprocessing.preprocessor import Preprocessor


class PreprocessingPipeline(Preprocessor):
    def __init__(self, column_preprocessors: Dict[str, Preprocessor]) -> None:
        super().__init__()

        self.column_preprocessors = column_preprocessors

    def check_is_fitted(self) -> None:
        unfitted_preprocessors = []
        for preprocessor in self.column_preprocessors.values():
            try:
                preprocessor.check_is_fitted()
            except NotFittedError:
                unfitted_preprocessors.append(preprocessor)
        if unfitted_preprocessors:
            raise NotFittedError(
                f"The column preprocessors "
                f"{[type(p).__name__ for p in unfitted_preprocessors]} "
                f"are not fitted yet. "
                f"Call 'fit' with appropriate arguments first"
            )

    def fit(self, df: pd.DataFrame) -> None:
        for col, preprocessor in self.column_preprocessors.items():
            preprocessor.fit(data=df[col].to_numpy().reshape(-1, 1))

    def transform(
        self,
        df: pd.DataFrame,
        only_cols: Optional[List[str]] = None,
        passthrough_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        self.check_is_fitted()

        df_transformed = pd.DataFrame()
        df_transformed.index = df.index

        for col, preprocessor in self.column_preprocessors.items():
            if only_cols is not None and col not in only_cols:
                continue
            data = preprocessor.transform(data=df[col].to_numpy().reshape(-1, 1))
            df_transformed[preprocessor.get_new_col_name(col_name=col)] = data

        if passthrough_cols:
            for col in passthrough_cols:
                df_transformed[col] = df[col]

        assert (df.index == df_transformed.index).all()

        return df_transformed

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.check_is_fitted()

        # this doesn't work because of the unknown column names of the inverse
        # transformed data

        # df_inverse_transformed = pd.DataFrame()
        # df_inverse_transformed.index = df.index
        #
        # for col, preprocessor in self.column_preprocessors.items():
        #     data = preprocessor.inverse_transform(data=df[col])
        #     df_inverse_transformed[preprocessor.get_new_col_name(col_name=col)] = data
        #
        # assert (df.index == df_inverse_transformed.index).all()
        #
        # return df_inverse_transformed

        raise NotImplementedError

    def get_new_col_name(self, col_name: Union[str, List[str]]) -> List[str]:
        self.check_is_fitted()

        if isinstance(col_name, str):
            # get the preprocessor responsible for the column `col_name`
            preprocessor = self.column_preprocessors[col_name]
            new_col_names = preprocessor.get_new_col_name(col_name=col_name)
        elif isinstance(col_name, list):
            col_names = col_name
            new_col_names = [
                col
                for col_name in col_names
                for col in self.get_new_col_name(col_name=col_name)
            ]  # to unpack the lists already
        else:
            raise ValueError(
                "col_name has to be and instance of either str or list, "
                f"was of type {type(col_name)}"
            )

        return new_col_names

    def save(self, path: pathlib.Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: pathlib.Path) -> PreprocessingPipeline:
        with open(path, "rb") as f:
            preprocessing_pipeline = pickle.load(f)
        return preprocessing_pipeline

    def __repr__(self):
        """Overrides the default implementation for representation."""
        return repr(self.column_preprocessors)

    def __eq__(self, other):
        """Overrides the default implementation for equality."""
        if isinstance(other, PreprocessingPipeline):
            return self.column_preprocessors == other.column_preprocessors
        return NotImplemented
