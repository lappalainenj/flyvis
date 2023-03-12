"""Utility on tensors and arrays."""
from typing import Any, Dict, Mapping
from numpy.typing import NDArray

import torch
import numpy as np
import pandas as pd

# -- Parameter dereferencing ---------------------------------------------------


class RefTensor:
    """Stores values and indices as tensors.

    Args:
        values (tensor)
        indices (tensor)

    Attributes: see Args.
    """

    def __init__(self, values: torch.Tensor, indices: torch.Tensor) -> None:
        self.values = values
        self.indices = indices

    def deref(self) -> torch.Tensor:
        """Indexes the values with the given indices in the last dimension."""
        return self.values.index_select(-1, self.indices)

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return "\nValues:\n{}\nReferences:\n{}\nDereferenced:\n{}".format(
            self.values, self.indices, self.deref()
        )

    @staticmethod
    def get_ref_indices(dataframe, groupby, group=pd.core.groupby.GroupBy.first):
        """Returns the grouped dataframe and the reference indices for shared
        parameters.

        Args:
            dataframe (pd.DataFrame): dataframe of nodes or edges, can also
                    contain synapse counts and signs.
            groupby (list): list of columns to group the dataframe by.
            group (method): groupby method, e.g. first, mean, sum.

        Returns:
            pd.DataFrame: first entries per group.
            tensor: indices for parameter sharing
        """
        group = group(dataframe.groupby(groupby, as_index=False, sort=False))
        ungrouped_elements = zip(*[dataframe[k][:] for k in groupby])
        grouped_elements = zip(*[group[k][:] for k in groupby])
        to_index = {k: i for i, k in enumerate(grouped_elements)}
        return group, torch.tensor([to_index[k] for k in ungrouped_elements])

    @staticmethod
    def scatter_mean(tensor, indices):
        """Scatter mean."""
        return scatter_mean(tensor, indices)

    @staticmethod
    def scatter_add(tensor, indices):
        """Scatter mean."""
        return scatter_add(tensor, indices)

    def clone(self):
        return RefTensor(self.values.clone(), self.indices)

    def detach(self):
        return RefTensor(self.values.detach(), self.indices)


class AutoDeref(dict):
    """
    An auto-dereferencing namespace.

    Note: dereferencing - obtain data address from pointer.
    Here, if attributes are RefTensors, __getitem__ will "deref"
    the tensor with the given indices.

    The cache speeds up processing if the same
    parameter is referenced multiple times in the dynamics.
    Is constructed in network._param_api(), i.e. cache is destroyed
    at each forward call.
    """

    _cache: Dict[str, object]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_cache", {})

    def __setitem__(self, key: str, value: object) -> None:
        self._cache.pop(key, None)
        super().__setitem__(key, value)

    def __getitem__(self, key: str) -> Any:
        try:
            val = super().__getitem__(key)
        except:
            raise AttributeError
        if isinstance(val, RefTensor):
            if key not in self._cache:
                self._cache[key] = val.deref()
            val = self._cache[key]
        return val

    def __setattr__(self, key: str, value: object) -> None:
        self.__setitem__(key, value)

    def __getattr__(self, key: str) -> Any:
        return self.__getitem__(key)

    def __repr__(self) -> str:
        def single_line_repr(elem: object) -> str:
            if isinstance(elem, list):
                return "[" + ", ".join(map(single_line_repr, elem)) + "]"
            elif isinstance(elem, AutoDeref):
                return (
                    f"{elem.__class__.__name__}("
                    + ", ".join(f"{k}={single_line_repr(v)}" for k, v in elem.items())
                    + ")"
                )
            else:
                return repr(elem).replace("\n", " ")

        def repr_in_context(elem: object, curr_col: int, indent: int) -> str:
            sl_repr = single_line_repr(elem)
            if len(sl_repr) <= 80 - curr_col:
                return sl_repr
            elif isinstance(elem, list):
                return (
                    "[\n"
                    + " " * (indent + 2)
                    + (",\n" + " " * (indent + 2)).join(
                        repr_in_context(e, indent + 2, indent + 2) for e in elem
                    )
                    + "\n"
                    + " " * indent
                    + "]"
                )
            elif isinstance(elem, AutoDeref):
                return (
                    f"{elem.__class__.__name__}(\n"
                    + " " * (indent + 2)
                    + (",\n" + " " * (indent + 2)).join(
                        f"{k} = " + repr_in_context(v, indent + 5 + len(k), indent + 2)
                        for k, v in elem.items()
                    )
                    + "\n"
                    + " " * indent
                    + ")"
                )
            else:
                return repr(elem)

        return repr_in_context(self, 0, 0)

    def get_as_reftensor(self, key):
        return dict.__getitem__(self, key)

    def clear_cache(self):
        object.__setattr__(self, "_cache", {})

        return clone(self)

    def detach(self):
        return detach(self)


def detach(obj: AutoDeref) -> AutoDeref:
    """
    Recursively detach AutoDeref mappings.
    """
    if isinstance(obj, (type(None), bool, int, float, str, type)):
        return obj
    elif isinstance(obj, (RefTensor, torch.Tensor)):
        return obj.detach()
    elif isinstance(obj, (list, tuple)):
        return [detach(v) for v in obj]
    elif isinstance(obj, Mapping):
        return AutoDeref({k: detach(dict.__getitem__(obj, k)) for k in obj})
    else:
        try:
            return detach(vars(obj))
        except TypeError as e:
            raise TypeError(f"{obj} of type {type(obj)} as {e}.")


def clone(obj: AutoDeref) -> AutoDeref:
    """
    Recursively clone AutoDeref mappings.
    """
    if isinstance(obj, (type(None), bool, int, float, str, type)):
        return obj
    elif isinstance(obj, (RefTensor, torch.Tensor)):
        return obj.clone()
    elif isinstance(obj, (list, tuple)):
        return [clone(v) for v in obj]
    elif isinstance(obj, Mapping):
        return AutoDeref({k: clone(dict.__getitem__(obj, k)) for k in obj})
    else:
        try:
            return clone(vars(obj))
        except TypeError as e:
            raise TypeError(f"{obj} of type {type(obj)} as {e}.")


def to_numpy(array):
    if isinstance(array, np.ndarray):
        return array
    elif isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    elif isinstance(array, list):
        return np.array(array)
    else:
        raise ValueError


def matrix_mask_by_sub(sub_matrix, matrix) -> NDArray[bool]:
    """Mask of rows in matrix that are contained in sub_matrix.

    Args:
        sub_matrix (array): shape (#rows1, #columns)
        matrix (array): shape (#rows2, #columns)

    Returns:
        array: 1D boolean array of length #rows2

    Note: #rows1 !<= #rows2

    Example:
        sub_matrix = np.array([[1, 2, 3],
                                [4, 3, 1]])
        matrix = np.array([[3, 4, 1],
                            [4, 3, 1],
                            [1, 2, 3]])
        matrix_mask_by_sub(sub_matrix, matrix)
        array([False, True, True])

    Typically, indexing a tensor with indices instead of booleans is
    faster. Therefore, see also where_equal_rows.
    """
    from functools import reduce

    n_rows, n_columns = sub_matrix.shape
    n_rows2 = matrix.shape[0]
    if not n_rows <= n_rows2:
        raise ValueError
    row_mask = []
    for i in range(n_rows):
        column_mask = []
        for j in range(n_columns):
            column_mask.append(sub_matrix[i, j] == matrix[:, j])
        row_mask.append(reduce(np.logical_and, column_mask))
    return reduce(np.logical_or, row_mask)


def where_equal_rows(matrix1, matrix2, as_mask=False) -> NDArray[int]:
    """Indices where matrix1 rows are in matrix2.

    Example:
        matrix1 = np.array([[1, 2, 3],
                            [4, 3, 1]])
        matrix2 = np.array([[3, 4, 1],
                            [4, 3, 1],
                            [1, 2, 3],
                            [0, 0, 0]])
        where_equal_rows(matrix1, matrix2)
        array([2, 1])
        matrix2[where_equal_rows(matrix1, matrix2)]
        array([[1, 2, 3],
               [4, 3, 1]])

    See also: matrix_mask_by_sub.
    """
    n_rows1 = matrix1.shape[0]
    n_rows2 = matrix2.shape[0]
    if not n_rows1 <= n_rows2:
        raise ValueError

    where = []
    rows = np.arange(matrix2.shape[0])
    for row in matrix1:
        equal_rows = (row == matrix2).all(axis=1)
        for index in rows[equal_rows]:
            where.append(index)
    return np.array(where)


# def ipython_variables_of_type(_globals, _type=torch.Tensor):
#     """To find the stale variables in a notebook to free gpu mem"""
#     stale = []
#     declared = []
#     for k, v in _globals.items():
#         if isinstance(v, _type):
#             if k.startswith("_"):
#                 stale.append(k)
#             else:
#                 declared.append(k)
#     return stale, declared


# def delete_stale_ipython_variables_of_type(_globals, _type=torch.Tensor):
#     """Actually deleting the stale variables in a notebook to free gpu mem"""
#     import gc

#     stale, declared = ipython_variables_of_type(_globals, _type)
#     for key in stale:
#         del _globals[key]
#     gc.collect()
#     torch.cuda.empty_cache()


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_reduce(src, index, dim=-1, mode="mean"):
    index = broadcast(index.long(), src, dim)
    return torch.scatter_reduce(src, dim, index, reduce=mode)


def scatter_mean(src, index, dim=-1):
    return scatter_reduce(src, index, dim, "mean")


def scatter_add(src, index, dim=-1):
    return scatter_reduce(src, index, dim, "sum")
