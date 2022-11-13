"""
Defines Paths class used across the package to handle filepaths.
"""
import pathlib


class Paths:
    """Class to reliably handle file paths relative to the root directory path.

    Attributes:
        root_path: The root path of the directory, which serves as the basis for all dynamically created filepaths.
    """

    def __init__(self) -> None:
        """Initializes Paths instance.

        Finds the root path as the directory two levels upwards of where this file is located.
        Prints out the detected root path.
        """
        self.root_path = pathlib.Path(__file__).resolve().parent.parent
        #print("self.root_path: ",self.root_path)

    @staticmethod
    def safe_return(path: pathlib.Path, path_type: str, mkdir: bool) -> pathlib.Path:
        """Safely return a path by optionally creating the parent directories to avoid errors when writing to the path.

        Args:
            path: Path to optionally create and return.
            path_type: Whether the path points to a directory or file
                (relevant for creating the path or parent path respectively).
            mkdir: If True, creates the parent directories. If False, it has no effect.

        Returns:
            Input path.
        """
        if mkdir:
            if path_type == "file":
                path.parent.mkdir(parents=True, exist_ok=True)
            elif path_type == "directory":
                path.mkdir(parents=True, exist_ok=True)
            else:
                raise ValueError(
                    f"'path_type' has to be either 'file' or 'directory'. "
                    f"Got: {path_type}"
                )
        print("path is ",path)  #/work/krgedia/CMSSW_10_1_0/src/Xbb/python/gnn_b_tagging_efficiency/configs/foo/foo.py      
        return path

    def config(self, config_type: str, config_name: str, mkdir: bool) -> pathlib.Path:
        """Returns path to a config file.

        Args:
            config_type: Type of the config file (e.g. 'dataset', ...).
            config_name: Name of the config file without the file extension.
            mkdir: If True, creates parent directories for safe writing. If False, won't create parents.

        Returns:
            Path to config file.
        """
        return self.safe_return(
            self.root_path / "configs" / config_type / f"{config_name}.py",
            path_type="file",
            mkdir=mkdir,
        )

    def extracted_dataset_dir(
        self, dataset_name: str, run_id: str, mkdir: bool
    ) -> pathlib.Path:
        """Returns path to the directory of output files of a dataset extraction.

        Args:
            dataset_name: Name of the dataset.
            run_id: Identifier of the run.
            mkdir: If True, creates parent directories for safe writing. If False, won't create parents.

        Returns:
            Path to the directory of output files of a dataset extraction.
        """
        return self.safe_return(
            self.root_path / "data" / dataset_name / "extractions" / run_id,
            path_type="directory",
            mkdir=mkdir,
        )

    def model_files_dir(
        self,
        dataset_name: str,
        dataset_handling_name: str,
        working_points_set_name: str,
        model_name: str,
        run_id: str,
        mkdir: bool,
    ) -> pathlib.Path:
        """Returns path to the directory of output files of a model on a dataset.

        Args:
            dataset_name: Name of the dataset.
            dataset_handling_name: Name of the dataset handling config.
            working_points_set_name: Name of the working points set config.
            model_name: Name of the model.
            run_id: Identifier of the run.
            mkdir: If True, creates parent directories for safe writing. If False, won't create parents.

        Returns:
            Path to the directory of output files of a model on a dataset.
        """
        return self.safe_return(
            self.root_path
            / "data"
            / dataset_name
            / dataset_handling_name
            / working_points_set_name
            / "models"
            / model_name
            / run_id,
            path_type="directory",
            mkdir=mkdir,
        )

    def evaluation_files_dir(
        self,
        dataset_name: str,
        dataset_handling_name: str,
        working_points_set_name: str,
        evaluation_name: str,
        run_id: str,
        working_point_name: str,
        mkdir: bool,
    ) -> pathlib.Path:
        """Returns path to the directory of evaluation files.

        Args:
            dataset_name: Name of the dataset.
            dataset_handling_name: Name of the dataset handling config.
            working_points_set_name: Name of the working points set config.
            evaluation_name: Name of the evaluation.
            run_id: Identifier of the run.
            working_point_name: Name of the working point.
            mkdir: If True, creates parent directories for safe writing. If False, won't create parents.

        Returns:
            Path to the directory of evaluation files.
        """
        return self.safe_return(
            self.root_path
            / "data"
            / dataset_name
            / dataset_handling_name
            / working_points_set_name
            / "evaluation"
            / evaluation_name
            / run_id
            / working_point_name,
            path_type="directory",
            mkdir=mkdir,
        )

    def dataset_outputs_dir(
        self,
        dataset_name: str,
        mkdir: bool,
    ) -> pathlib.Path:
        """Returns path to the directory of dataset output files.

        Args:
            dataset_name: Name of the dataset.
            mkdir: If True, creates parent directories for safe writing. If False, won't create parents.

        Returns:
            Path to the directory of dataset output files.
        """
        return self.safe_return(
            self.root_path / "data" / dataset_name / "outputs",
            path_type="directory",
            mkdir=mkdir,
        )
