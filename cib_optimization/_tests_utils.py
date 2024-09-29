import mlflow
import numpy as np
import torch
from scipy.stats import entropy
from torch import Tensor

from _probability_utils import rvs_einsum


def variation_of_information_of_abstractions(qTcondX, qRcondX, pX, NTs, NRs, NXs):
    """Compute the variation of information of abstractions T and R."""
    var_numbers = {
        "t": len(NTs),
        "r": len(NRs),
        "x": len(NXs),
    }
    # Compute the joint and the marginals
    qTR: Tensor = rvs_einsum(
        (qTcondX, qRcondX, pX), ("tx", "rx", "x"), "tr", var_numbers
    )
    qT: Tensor = rvs_einsum((qTcondX, pX), ("tx", "x"), "t", var_numbers)
    qR: Tensor = rvs_einsum((qRcondX, pX), ("rx", "x"), "r", var_numbers)

    # Mutual Information
    MI: float = mutual_information(qTR, qT, qR)

    # Entropies
    HT: float = entropy(qT, base=2)
    HR: float = entropy(qR, base=2)

    # Variation of Information
    VI = HT + HR - 2 * MI

    return VI


def mutual_information(p_xy, p_x, p_y, eps=1e-10) -> float:
    """Calculate the mutual information between two probability distributions.

    Args:
        p_xy: Joint distribution.
        p_x: First probability distribution.
        p_y: Second probability distribution.
        eps: A small constant to avoid division by zero (default: 1e-10).
    """
    p_x_expand = p_x.unsqueeze(1)  # Expand p_x along the second dimension
    p_y_expand = p_y.unsqueeze(0)  # Expand p_y along the first dimension
    p_xy_product = torch.mul(p_x_expand, p_y_expand)  # Compute the product P_X ⊗ P_Y

    summand = torch.where(
        torch.logical_and(p_xy_product > eps, p_xy / p_xy_product > eps),
        p_xy * torch.log2(p_xy / p_xy_product),
        torch.zeros_like(p_xy),
    )

    # Mutual information: I(X;Y) = KL(P_{XY} || P_X ⊗ P_Y)
    return float(torch.sum(summand))


def project_onto_prob_simplex(V):
    """V is matrix with one vector to project on each row."""
    N, D = V.shape

    MU = np.sort(V, axis=1)[:, ::-1]  # Sort Y in descending order along axis=1

    MUcum = (np.cumsum(MU, axis=1) - 1) @ np.diag(1.0 / np.arange(1, D + 1))
    mask = MU > MUcum
    col_idxs = np.arange(1, MU.shape[1] + 1)
    masked_col_idxs_array = np.tile(col_idxs, (MU.shape[0], 1)) * mask
    rho = np.max(masked_col_idxs_array, axis=1)

    theta = np.expand_dims(1 / rho * (np.sum(MU * mask, axis=1) - 1.0), axis=-1)

    W = np.maximum(V - theta, 0)

    return W


# Initialize by sampling uniformly from simplex
def simplex_uniform_sampling(n: int) -> Tensor:
    """n is the dimension of the simplex."""
    # Generate n-1 random numbers in the interval (0, 1) using PyTorch
    random_numbers = torch.rand(n - 1)

    # Add 0 and 1 to the list
    random_numbers_padded = torch.cat((random_numbers, torch.tensor([0.0, 1.0])))

    # Sort the list
    random_numbers_padded, _ = torch.sort(random_numbers_padded)

    # Record the differences between consecutive elements
    differences = random_numbers_padded[1:] - random_numbers_padded[:-1]

    return differences


def log_metrics(loss, diff_q, diff_loss, components, iteration):
    mlflow.log_metric("CIB loss", loss.item(), step=iteration)
    mlflow.log_metric("diff_q", diff_q, step=iteration)
    mlflow.log_metric("diff_loss", diff_loss, step=iteration)
    mlflow.log_metric("HT", components["HT"], step=iteration)
    mlflow.log_metric("HTcondX", components["HTcondX"], step=iteration)
    mlflow.log_metric("HY", components["HY"], step=iteration)
    mlflow.log_metric("HcYdoT", components["HcYdoT"], step=iteration)
    mlflow.log_metric("penalty", components["penalty"], step=iteration)


def monitor_iteration(
    iteration: int,
    optimizer,
    loss,
    diff_q: float,
    diff_loss: float,
    components: dict[str, float],
):
    """Print the monitoring details for the current iteration."""
    HT = components["HT"]
    HTcondX = components["HTcondX"]
    HY = components["HY"]
    HcYdoT = components["HcYdoT"]
    penalty = components["penalty"]

    # Print monitoring information
    print(
        f"\n\tloss: {loss.item()}"
        + f"\n\tlr: {optimizer.param_groups[0]['lr']}"
        + f"\n\tDiff_loss: [{diff_loss}]"
        + f"\n\tDiff_q: [{diff_q}]"
        + f"\n\n\tq: [{optimizer.param_groups[0]['params'][0]}]"
    )
    print(
        f"\n\tHT = {HT}\n\tHTcondX = {HTcondX}\n\tHY = {HY}\n\tHcYdoT = {HcYdoT}"
        + f"\n\tpenalty = {penalty}"
    )


def detect_cycling(q, q_last1, q_last2, loss_last1, loss_last2, new_loss):
    """Detect if q is cycling between the last two values and returns the best q.

    Args:
    - q (Tensor): Current tensor value of q.
    - q_last1 (Tensor): Last tensor value of q.
    - q_last2 (Tensor): Second last tensor value of q.
    - loss_last1 (float): Loss corresponding to q_last1.
    - loss_last2 (float): Loss corresponding to q_last2.
    - new_loss (float): Loss corresponding to the current q.

    Returns:
    - best_q (Tensor): The best tensor value of q based on the lowest loss.
    - cycling_detected (bool): True if cycling is detected, False otherwise.
    """
    if q_last1 is not None and q_last2 is not None:
        if torch.equal(q, q_last2):
            print("Cycling detected between two values of q.")
            # Choose the q with the best (lowest) loss
            if loss_last1 < loss_last2:
                return q_last1, True
            elif loss_last2 <= loss_last1:
                return q_last2, True
    return None, False


def update_history(q, loss, q_last1, q_last2, loss_last1, loss_last2):
    """
    Update the history of q values and losses for the next iteration.

    Args:
    - q (Tensor): Current tensor value of q.
    - q_last1 (Tensor): Last tensor value of q.
    - q_last2 (Tensor): Second last tensor value of q.
    - loss_last1 (float): Loss corresponding to q_last1.
    - loss_last2 (float): Loss corresponding to q_last2.

    Returns:
    - q_last1 (Tensor): Updated last tensor value of q.
    - q_last2 (Tensor): Updated second last tensor value of q.
    - loss_last1 (float): Updated loss corresponding to q_last1.
    - loss_last2 (float): Updated loss corresponding to q_last2.
    """
    q_last2 = q_last1
    q_last1 = q.clone().detach()

    loss_last2 = loss_last1
    loss_last1 = loss.item()  # Update with the new loss

    return q_last1, q_last2, loss_last1, loss_last2


def permute_first_indices(t: Tensor, n: int) -> Tensor:
    total_dims = t.dim()
    # First n indices to end
    new_order = list(range(n, total_dims)) + list(range(n))
    # Reorder the dimensions
    permuted_t = t.permute(*new_order)
    return permuted_t
