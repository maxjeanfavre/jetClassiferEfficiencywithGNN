"""
Custom warnings and error classes used across the package.
"""


class Error(Exception):
    """Base class for exceptions in this package."""


class UnexpectedError(Error):
    """Exception class to raise if an unexpected error occurs."""


class NotFittedError(Error):
    """Exception class to raise if an estimator is used before fitting."""


class TTreeBranchesRowLengthsMismatchError(Error):
    """Exception class to raise when TTree branches row lengths are incompatible."""
