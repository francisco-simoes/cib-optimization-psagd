"""Useful function to manipulate probability distributions/tensors."""

import numpy as np
import torch
from torch import Tensor

LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def marginal(p: np.ndarray, xs: list, squeeze_dims: bool = True) -> np.ndarray:
    """Compute marginal distribution of xs from joint p.

    If squeeze_dims is False, shape or returned array has 1s in
    summed over dimensions.
    """
    sum_over: list = [j for j in range(p.ndim) if j not in xs]
    marg = np.apply_over_axes(np.sum, p, sum_over)
    if squeeze_dims:
        marg = np.squeeze(marg)
    return marg


def conditionals(
    p: np.ndarray,
    xs: list,
    cond_set: list,
    conditioned_last: bool = True,
    eps: float = 1e-10,
) -> np.ndarray:
    """Compute the conditional distributions.

    Say xs = [0, 2] corresponding to X and Z while
    cond_set = [1, 3] corresponding to Y, W. Then
    conditionals(p, xs, cond_set)[:, :, 7, 2] corresponds to the
    conditional distribution p(x, z | y=7, w=2).
    """
    # Do not squeeze: shape with 1s on the xs for broadcasting to work as desired
    numerator = marginal(p, xs + cond_set, squeeze_dims=False)
    denominator = marginal(p, cond_set, squeeze_dims=False)
    # cond_dists = np.divide(numerator, denominator)
    cond_dists = np.where(
        denominator > eps,
        np.divide(numerator, denominator),
        np.zeros(numerator.shape),
    )

    if conditioned_last:  # Conditioned variables in last dimensions
        cond_dists = _move_dimensions_to_last(cond_dists, cond_set)
        # Won't need collapsed dimensions in this case
        cond_dists = np.squeeze(cond_dists)

    return cond_dists


def _move_dimensions_to_last(
    a: np.ndarray, dimensions_to_move_last: list[int]
) -> np.ndarray:
    """
    Move specified dimensions to the last positions in the given array.

    Parameters:
    - array (numpy.ndarray): The input array.
    - dimensions_to_move_last (list[int]): list of dimensions to move last.

    Returns:
    numpy.ndarray: An array with specified dimensions moved last.
    """
    current_dims_order = list(range(a.ndim))

    # Move specified dimensions
    new_dims_order = [
        dim for dim in current_dims_order if dim not in dimensions_to_move_last
    ] + dimensions_to_move_last
    # reordered = np.moveaxis(a, current_dims_order, new_dims_order)
    reordered = np.transpose(a, new_dims_order)

    return reordered


def rvs_einsum(
    tensors: tuple[Tensor],
    tensors_rvs_indices: tuple[str],
    final_rvs_indices: str,
    var_numbers: dict[str, int],
) -> Tensor:
    """
    Perform an Einstein summation operation on tensors indexed by random vectors.

    This function enables the computation of einsum operations over tensors
    whose indices can be grouped in different types, represented by
    single letters (e.g., "x"). This is the case, for instance, when the
    indices correspond to random vectors.
    For example, to compute the einsum operation represented by:
        x1 x2 z1, x1 x2 t1 -> z1 t1
    you can express it using:
        "xz,xt->zt"
    where `tensors_rvs_indices` would be ("xz", "xt") and
    `final_rvs_indices` would be "zt".

    Args:
        tensors (tuple[Tensor]): A tuple of tensors to be operated on.
        tensors_rvs_indices (tuple[str]): A tuple of strings representing the
            indices for each tensor in the einsum operation.
        final_rvs_indices (str): A string representing the indices in the final result.
        var_numbers (dict[str, int]): A dictionary mapping index types to
            their respective numerical sizes.

    Returns:
        Tensor: The result of the einsum operation, combining the input tensors
        according to the specified indices.

    Raises:
        ValueError: If the provided tensor indices do not match the expected
        shapes or if the einsum operation is not valid.

    Example:
        >>> result = rvs_einsum((t1, t2), ("xz", "xt"), "zt", {"x": 2, "z": 3, "t": 4})

        This performs einsum for tensors t1, t2 with 2 x-indices, 3 z-indices
        and 4 t-indices. In the context of probability distributions, t1 may
        be a conditional distribution p(x|z), while t2 may be a joint
        distribution p(x,t), for example. The result of this operation would
        then be a tensor t3 with t3[z,t] = sum_{x} p(x|z)p(x,t).
    """
    assert len(tensors) == len(tensors_rvs_indices)

    # Build dict mapping rvs to their indices
    rv_to_indices: dict[str, list[int]] = {}  # example "t": [2,3,4]; defined using a
    # first, then adding missing ones from b.
    rvs_in_order: str = ""  # example: "xtz"
    for string in tensors_rvs_indices:
        for rv in string:
            if rv not in rvs_in_order:
                rvs_in_order += rv
    total_numerical_indices = 0
    for var in rvs_in_order:
        var_number = var_numbers[var]  # Number of current var (e.g. 2 if X1,X2)
        rv_to_indices[var] = list(
            range(total_numerical_indices, total_numerical_indices + var_number)
        )
        total_numerical_indices += var_number

    # Build collection to feed to einsum
    tensors_sublists_collection: tuple = ()
    # (Of form (tensor1, [indices of tensor1], tensor2, [indices of tensor2, ...]))
    for i, tensor in enumerate(tensors):
        indices_string = tensors_rvs_indices[i]
        indices: list[int] = []
        for rv in indices_string:
            indices += rv_to_indices[rv]
        tensors_sublists_collection += (tensor, indices)
    final_indices: list[int] = []
    for var in final_rvs_indices:
        final_indices += rv_to_indices[var]

    result = torch.einsum(*tensors_sublists_collection, final_indices)

    return result
