"""Defines the DatasetConfig class."""
from typing import Dict, List, Optional, Tuple

from utils.configs.config import Config


class DatasetConfig(Config):
    """Container for information about a dataset.

    Attributes:
        name: The name of the dataset.
        path: The path to the dataset.
        key: The key of the TTree in the ROOT file.
        file_limit: Upper limit of files to read in.
        filename_pattern: Pattern of the filenames.
        branches_to_simulate: If not None, will create corresponding columns based on
            the given expressions. E.g.: ({"name": "Jet_Pt", "expressions": "Jet_pt"},)
            will create the Jet_Pt column using the values from Jet_pt.
        filenames: Filenames of the ROOT files.
    """

    def __init__(
        self,
        name: str,
        path: str,
        key: str,
        file_limit: Optional[int],
        filename_pattern: str,
        branches_to_simulate: Optional[List[Dict[str, str]]],
        filenames: Tuple[str, ...],
    ) -> None:
        if path[-1] != "/":
            raise ValueError(f"Dataset path has to end with '/'. Got: {path}")

        super().__init__(name=name)

        self.path = path
        self.key = key
        self.file_limit = file_limit
        self.filename_pattern = filename_pattern
        self.branches_to_simulate = branches_to_simulate
        self.filenames = filenames
