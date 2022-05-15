from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


class Histogram:
    def __init__(
        self,
        h: np.ndarray,
        edges: Tuple[np.ndarray, ...],
        variables: Tuple[str, ...],
    ):
        self._h = None
        self._edges = None
        self._variables = None

        self.edges = tuple(np.copy(i) for i in edges)
        self.variables = tuple(variables)
        self.h = np.copy(h)

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise ValueError("Value for h must be an instance of 'np.ndarray'")
        if value.ndim != len(self.edges):
            raise ValueError(
                f"Dimensions of value for h ({value.ndim}) are not compatible "
                f"with the length of edges ({len(self.edges)})"
            )
        if value.shape != tuple(len(i) - 1 for i in self.edges):
            raise ValueError(
                "Shape of h is incompatible with the 'edges' attribute. "
                f"The value for h has shape {value.shape}, while the 'edges' "
                f"attribute has entries of length {tuple(len(i) for i in self.edges)} "
                f"which requires h to be of shape "
                f"{(tuple(len(i) - 1 for i in self.edges))}"
            )

        self._h = value

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, value: Tuple[np.ndarray, ...]):
        if self.h is not None:
            raise ValueError("The attribute 'h' has to be set first")
        if not isinstance(value, tuple):
            raise ValueError(
                "Value for edges must be a tuple of 'np.ndarray' instances"
            )
        if any(not isinstance(i, np.ndarray) for i in value):
            raise ValueError(
                "Value for edges must be a tuple of 'np.ndarray' instances. "
                f"The tuple entries had types {[type(i) for i in value]}"
            )
        if any(i.ndim != 1 for i in value):
            raise ValueError(
                "Value for edges has to be a tuple of 1D np.ndarray. "
                f"The tuple entries had shapes: {[i.shape for i in value]}"
            )
        if any(len(i) < 2 for i in value):
            raise ValueError(
                "The value for edges had entries with length below 2. "
                "For the minimum of one bin, there should be at least 2 bin edges. "
                f"The entries had lengths: {[len(i) for i in value]}"
            )
        if self.variables is not None and len(self.variables) != len(value):
            raise ValueError(
                "Lengths of value for edges and 'variable' attributes not equal. "
                f"Value for edges had length {len(value)}, "
                f"the 'variables' attribute had length {len(self.variables)}"
            )
        if not all(np.all(np.diff(e) > 0) for e in value):
            raise ValueError(
                f"The entries of edges have to be strictly increasing. Got: {value}"
            )
        self._edges = value

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, value: Tuple[str, ...]):
        if not isinstance(value, tuple):
            raise ValueError("Value for variables must be a tuple of 'str' instances")
        if any(not isinstance(i, str) for i in value):
            raise ValueError(
                "Value for variables must be a tuple of 'str' instances. "
                f"The tuple entries had types {[type(i) for i in value]}"
            )
        if self.edges is not None and len(self.edges) != len(value):
            raise ValueError(
                "Lengths of value for variables and 'edges' attributes not equal. "
                f"Value for variables had length {len(value)}, "
                f"the 'edges' attribute had length {len(self.edges)}"
            )
        if len(set(value)) != len(value):
            raise ValueError(
                "Value for variables has to be a tuple with unique entries. "
                f"Had entries: {value}"
            )
        self._variables = value

    @classmethod
    def from_df_and_bins(cls, df: pd.DataFrame, bins) -> Histogram:
        data = df[bins.keys()].to_numpy()
        h, edges = np.histogramdd(
            sample=data,
            bins=list(bins.values()),
        )

        hist = cls(h=h, edges=tuple(edges), variables=tuple(bins.keys()))

        return hist

    def get_bin_entry(self, df: pd.DataFrame) -> np.ndarray:
        bin_indices = df[list(self.variables)].apply(
            lambda x: np.digitize(
                x, self.edges[self.variables.index(x.name)], right=False
            )
        )  # these are the indices of the right edge of the bin the values fall into
        bin_indices -= 1  # now they are the bin indices
        bin_indices = bin_indices.to_numpy()
        bin_indices = bin_indices.T
        bin_indices = tuple(bin_indices)

        res = self.h[bin_indices]

        # sanity check
        bin_indices_2 = []
        for var, edges in zip(self.variables, self.edges):
            var_bin_indices = np.digitize(df[var], edges, right=False)
            var_bin_indices -= 1
            bin_indices_2.append(var_bin_indices)
        bin_indices_2 = tuple(bin_indices_2)
        assert len(bin_indices) == len(bin_indices_2)
        for i in range(len(bin_indices)):
            assert np.array_equal(bin_indices[i], bin_indices_2[i])
        res_2 = self.h[bin_indices_2]
        assert np.array_equal(res, res_2, equal_nan=True)
        # end sanity check

        return res

    def project(self, projection_variables) -> Histogram:
        # project histogram along projection variables
        # --> get histogram in the projection variables
        #     and other variables are projected out
        if not set(projection_variables).issubset(set(self.variables)):
            raise ValueError(
                "Requested projection_variables are not a "
                "subset of the variables of the histogram."
                "These variables were requested but are "
                "not variables of the histogram: "
                f"{set(projection_variables) - set(self.variables)}"
            )

        # old version
        # # this 'sorted' makes sure the match between variables, edges, and h is kept
        # idx_projection_variables = sorted(
        #     [self.variables.index(i) for i in projection_variables]
        # )
        # idx_all_variables = tuple(range(0, len(self.variables)))
        # idx_variables_to_sum = tuple(
        #     sorted(set(idx_all_variables) - set(idx_projection_variables))
        # )
        # # h = self.h.sum(axis=idx_variables_to_sum)
        # h = np.nansum(self.h, axis=idx_variables_to_sum)
        # edges = tuple(self.edges[i] for i in idx_projection_variables)
        # variables = tuple(self.variables[i] for i in idx_projection_variables)

        # new version
        idx_projection_variables = [
            self.variables.index(i) for i in projection_variables
        ]
        idx_all_variables = tuple(range(0, len(self.variables)))
        idx_variables_to_sum = tuple(
            sorted(set(idx_all_variables) - set(idx_projection_variables))
        )
        # this makes sure the match between variables, edges, and h is kept
        h_axes_moved = np.moveaxis(
            self.h, idx_projection_variables, sorted(idx_projection_variables)
        )
        h = np.nansum(h_axes_moved, axis=idx_variables_to_sum)
        edges = tuple(self.edges[i] for i in idx_projection_variables)
        variables = tuple(self.variables[i] for i in idx_projection_variables)

        projected_hist = Histogram(
            h=h,
            edges=edges,
            variables=variables,
        )
        return projected_hist

    def without_under_over_flow(self) -> Histogram:
        new_h = self.h[
            np.ix_(*[np.where(~np.isinf(np.diff(edges)))[0] for edges in self.edges])
        ]
        new_edges = tuple(edges[~np.isinf(edges)] for edges in self.edges)

        hist_without_under_over_flow = Histogram(
            h=new_h,
            edges=new_edges,
            variables=self.variables,
        )

        return hist_without_under_over_flow

    def __add__(self, other):
        """Overrides the default implementation for add."""
        if isinstance(other, int):
            h = self.h + other
            histogram = Histogram(h=h, edges=self.edges, variables=self.variables)
            return histogram
        else:
            return NotImplemented

    def __sub__(self, other):
        """Overrides the default implementation for sub."""
        if isinstance(other, Histogram):
            if (
                self.variables == other.variables
                and len(self.edges) == len(other.edges)
                and all(np.array_equal(i, j) for i, j in zip(self.edges, other.edges))
            ):
                h = self.h - other.h
                histogram = Histogram(h=h, edges=self.edges, variables=self.variables)
                return histogram
            else:
                return NotImplemented
        return NotImplemented

    def __mul__(self, other):
        """Overrides the default implementation for mul."""
        if isinstance(other, Histogram):
            if (
                self.variables == other.variables
                and len(self.edges) == len(other.edges)
                and all(np.array_equal(i, j) for i, j in zip(self.edges, other.edges))
            ):
                h = self.h * other.h
                histogram = Histogram(h=h, edges=self.edges, variables=self.variables)
                return histogram
            else:
                return NotImplemented
        return NotImplemented

    def __pow__(self, power, modulo=None):
        """Overrides the default implementation for pow."""
        h = pow(base=self.h, exp=power, mod=modulo)
        histogram = Histogram(h=h, edges=self.edges, variables=self.variables)
        return histogram

    def __truediv__(self, other):
        """Overrides the default implementation for truediv."""
        if isinstance(other, Histogram):
            if (
                self.variables == other.variables
                and len(self.edges) == len(other.edges)
                and all(np.array_equal(i, j) for i, j in zip(self.edges, other.edges))
                and self.h.shape == other.h.shape
            ):
                # h = self.h / other.h
                h = np.divide(
                    self.h,
                    other.h,
                    out=np.full(shape=self.h.shape, fill_value=np.nan),
                    where=other.h != 0,
                )
                histogram = Histogram(h=h, edges=self.edges, variables=self.variables)
                return histogram
            else:
                return NotImplemented
        return NotImplemented

    def __eq__(self, other):
        """Overrides the default implementation for equality."""
        if isinstance(other, Histogram):
            if (
                all((i == j).all() for i, j in zip(self.edges, other.edges))
                and self.variables == other.variables
                and np.array_equal(self.h, other.h, equal_nan=True)
            ):
                return True
            else:
                return False
        return NotImplemented

    def __repr__(self):
        """Overrides the default implementation for representation."""
        return f"Histogram(h={self.h}, edges={self.edges}, variables={self.variables})"
